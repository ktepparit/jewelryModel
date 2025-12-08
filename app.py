import streamlit as st
import json
import requests
import base64
from io import BytesIO
from PIL import Image
import time
import pandas as pd
import re

# --- 1. CONFIGURATION & CONSTANTS ---
st.set_page_config(layout="wide", page_title="Jewelry AI Studio")

# แก้ไขชื่อโมเดลให้ถูกต้อง (Gemini 3 ยังไม่เปิดใช้ทั่วไป ให้ใช้ 1.5 flash หรือ pro)
MODEL_IMAGE_GEN = "models/gemini-1.5-flash" 
MODEL_TEXT_SEO = "models/gemini-1.5-flash"

# --- HELPER: CLEANER (ตัวแก้บั๊กหลัก) ---
def clean_key(value):
    """ฟังก์ชันล้างค่า: ลบฟันหนู ลบช่องว่าง และ Newline ออกให้หมด"""
    if value is None: return ""
    return str(value).strip().replace(" ", "").replace('"', "").replace("'", "").replace("\n", "")

# Prompt A: สำหรับ Gen SEO (Tab 1)
SEO_PROMPT_POST_GEN = """
You are an SEO specialist with 15-20 years of experience. 
Help write SEO-optimized image file name with image alt tags in English for the product image with a model created, having product details according to this url: {product_url}
To rank well on organic search engines by customer groups interested in this type of product.

IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure:
{
  "file_name": "your-optimized-filename.jpg",
  "alt_tag": "Your optimized descriptive alt tag"
}
"""

# Prompt B: สำหรับ Bulk SEO (Tab 2)
SEO_PROMPT_BULK_EXISTING = """
You are an SEO specialist. Write SEO-optimized image file name and alt tags in English based on this url: {product_url}.
IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure:
{
  "file_name": "your-optimized-filename.jpg",
  "alt_tag": "Your optimized descriptive alt tag"
}
"""

# Default Data
DEFAULT_PROMPTS = [
    {
        "id": "p1", "name": "Luxury Hand (Ring)", "category": "Ring",
        "template": "A realistic close-up of a female hand model wearing a ring with {face_size} face size, soft studio lighting, elegant jewelry photography.",
        "variables": "face_size",
        "sample_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Ring_render.jpg/320px-Ring_render.jpg"
    },
    {
        "id": "p2", "name": "Streetwear Necklace", "category": "Necklace",
        "template": "A fashion portrait of a model wearing a {length} necklace, streetwear outfit, urban background, high detailed texture.",
        "variables": "length",
        "sample_url": "https://images.unsplash.com/photo-1611591437281-460bfbe1220a?q=80&w=300&auto=format&fit=crop"
    }
]

# --- 2. CLOUD DATABASE FUNCTIONS (JsonBin.io) ---
def get_prompts():
    try:
        # ดึงค่าและล้างค่าทันทีแก้ปัญหา Connection Adapter Error
        raw_key = st.secrets.get("JSONBIN_API_KEY", "")
        raw_bin = st.secrets.get("JSONBIN_BIN_ID", "")
        
        API_KEY = clean_key(raw_key)
        BIN_ID = clean_key(raw_bin)

        if not API_KEY or not BIN_ID:
            return DEFAULT_PROMPTS

        # URL ต้องสะอาด ไม่มีฟันหนู
        url = f"https://api.jsonbin.io/v3/b/{BIN_ID}/latest"
        headers = {"X-Master-Key": API_KEY}
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get("record", DEFAULT_PROMPTS)
        return DEFAULT_PROMPTS
    except Exception as e:
        # st.error(f"DB Error: {e}") # ปิด error ไว้ก่อนเพื่อไม่ให้รกหน้าจอถ้าโหลดไม่ได้
        return DEFAULT_PROMPTS

def save_prompts(data):
    try:
        raw_key = st.secrets.get("JSONBIN_API_KEY", "")
        raw_bin = st.secrets.get("JSONBIN_BIN_ID", "")
        
        API_KEY = clean_key(raw_key)
        BIN_ID = clean_key(raw_bin)
        
        url = f"https://api.jsonbin.io/v3/b/{BIN_ID}"
        headers = {"Content-Type": "application/json", "X-Master-Key": API_KEY}
        requests.put(url, json=data, headers=headers, timeout=10)
    except Exception as e:
        st.error(f"Save failed: {e}")

# --- 3. HELPER FUNCTIONS ---
def img_to_base64(img):
    buf = BytesIO()
    if img.mode == 'RGBA': img = img.convert('RGB')
    img.thumbnail((1024, 1024)) 
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def parse_json_response(text):
    try:
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text)
        text = text.strip()
        return json.loads(text)
    except:
        return None

# Function 1: Gen รูปภาพ
def generate_image(api_key, image_list, prompt):
    # ล้าง Key ก่อนใช้เสมอ
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_IMAGE_GEN}:generateContent?key={key}"
    
    parts = [{"text": f"Instruction: {prompt} \nConstraint: Keep the jewelry products in the input images EXACTLY as they are. Analyze all images to understand the 3D structure. Generate a realistic model wearing it."}]
    for img in image_list:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    try:
        res = requests.post(url, json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.3}}, headers={"Content-Type": "application/json"})
        if res.status_code != 200: return None, f"API Error {res.status_code}: {res.text}"
        
        content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
        if "inline_data" in content: return base64.b64decode(content["inline_data"]["data"]), None
        if "inlineData" in content: return base64.b64decode(content["inlineData"]["data"]), None
        if "text" in content: return None, f"Model returned text instead of image: {content['text']}"
        return None, "Unknown response format."
    except Exception as e: return None, str(e)

# Function 2: Gen SEO Post-Gen
def generate_seo_tags_post_gen(api_key, product_url):
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_TEXT_SEO}:generateContent?key={key}"
    final_seo_prompt = SEO_PROMPT_POST_GEN.replace("{product_url}", product_url)
    payload = {
        "contents": [{"parts": [{"text": final_seo_prompt}]}],
        "generationConfig": {"temperature": 0.5, "responseMimeType": "application/json"}
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                if "text" in content: return content["text"], None
                return None, "No text returned from model."
            elif res.status_code == 503:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                    continue 
                else: return None, "API Error 503: Overloaded."
            else: return None, f"API Error {res.status_code}: {res.text}"
        except Exception as e:
             if attempt < max_retries - 1: time.sleep(2); continue
             return None, str(e)
    return None, "Unknown error."

# Function 3: Bulk SEO
def generate_seo_for_existing_image(api_key, img_pil, product_url):
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_TEXT_SEO}:generateContent?key={key}"
    final_prompt = SEO_PROMPT_BULK_EXISTING.replace("{product_url}", product_url)
    
    payload = {
        "contents": [{
            "parts": [
                {"text": final_prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}}
            ]
        }],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 2048, "responseMimeType": "application/json"}
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                if "text" in content: return content["text"], None
                return None, "Model returned no text."
            elif res.status_code == 503 and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
                continue
            else: return None, f"API Error {res.status_code}: {res.text}"
        except Exception as e:
            if attempt < max_retries - 1: time.sleep(2); continue
            return None, str(e)
            
    return None, "Failed after retries."

def list_available_models(api_key):
    # ล้าง Key ก่อนเรียกเสมอ
    key = clean_key(api_key)
    # URL ต้องห้ามมีช่องว่างเด็ดขาด
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get("models", [])
        else:
            return None
    except:
        return None

# --- 4. UI LOGIC ---
if "library" not in st.session_state:
    st.session_state.library = get_prompts()
if "edit_target" not in st.session_state:
    st.session_state.edit_target = None
if "image_generated_success" not in st.session_state:
    st.session_state.image_generated_success = False
if "current_generated_image" not in st.session_state:
    st.session_state.current_generated_image = None

with st.sidebar:
    st.title("⚙️ Config")
    
    # 1. ลองดึงจาก Secrets ก่อน
    secret_key = st.secrets.get("GEMINI_API_KEY", "")
    
    # 2. ถ้ามีใน Secrets ให้ใช้เลย แต่ถ้าไม่มี ให้แสดงช่องกรอก
    if secret_key:
        api_key = secret_key
        st.success("API Key Ready (from Secrets)")
    else:
        api_key = st.text_input("Gemini API Key", type="password")
    
    # ล้างค่า Key ให้สะอาดก่อนส่งไปใช้ใน Global Variable
    api_key = clean_key(api_key)
    
    if "JSONBIN_API_KEY" in st.secrets:
        st.caption("✅ Database Connected")
    else:
        st.warning("⚠️ Local Mode")

st.title("💎 Jewelry AI Studio")

tab1, tab2, tab3, tab4 = st.tabs(["✨ Generate Image", "🏷️ Bulk SEO Tags", "📚 Library Manager", "ℹ️ About Models"])

# === TAB 1: GENERATE IMAGE ===
with tab1:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("1. Upload Reference")
        files = st.file_uploader("Upload Images for Gen", accept_multiple_files=True, type=["jpg", "png", "jpeg"], key="gen_upload")
        images_to_send = [Image.open(f) for f in files] if files else []
        if images_to_send:
            st.caption(f"Selected {len(images_to_send)} images:")
            cols = st.columns(4)
            for i, img in enumerate(images_to_send): cols[i%4].image(img, use_column_width=True)

    with c2:
        st.subheader("2. Settings")
        lib = st.session_state.library
        cats = list(set(p.get('category', 'Other') for p in lib)) if lib else []
        sel_cat = st.selectbox("Category", cats) if cats else None
        
        filtered = [p for p in lib if p.get('category') == sel_cat]
        if filtered:
            sel_style = st.selectbox("Style", filtered, format_func=lambda x: x.get('name', 'Unknown'))
            if sel_style.get("sample_url"): st.image(sel_style["sample_url"], width=100)
            
            vars_list = [v.strip() for v in sel_style.get('variables', '').split(",") if v.strip()]
            user_vals = {v: st.text_input(v, placeholder="e.g. Gold") for v in vars_list}
            
            final_prompt = sel_style.get('template', '')
            for k, v in user_vals.items(): final_prompt = final_prompt.replace(f"{{{k}}}", v)
            
            st.write("✏️ **Edit Prompt:**")
            prompt_edit = st.text_area("Instruction", value=final_prompt, height=100)
            
            if st.button("🚀 GENERATE IMAGE", type="primary", use_container_width=True):
                if not api_key or not images_to_send:
                    st.error("Check Key & Images")
                else:
                    with st.spinner(f"Generating Image ({MODEL_IMAGE_GEN})..."):
                        d, e = generate_image(api_key, images_to_send, prompt_edit)
                        if d:
                            st.session_state.current_generated_image = d
                            st.session_state.image_generated_success = True
                            st.rerun()
                        else:
                            st.error(e)
                            st.session_state.image_generated_success = False

            if st.session_state.image_generated_success and st.session_state.current_generated_image:
                st.divider()
                st.subheader("🎉 Result")
                st.image(st.session_state.current_generated_image, use_column_width=True)
                st.download_button("Download Image", st.session_state.current_generated_image, "gen.jpg", "image/jpeg", type="primary")
                
                st.divider()
                st.subheader("🌍 SEO Tools")
                product_url_input = st.text_input("Paste Product URL here:", key="post_gen_url")
                
                if st.button("✨ Gen Tags"):
                    if not product_url_input: st.warning("Please enter URL.")
                    else:
                        with st.spinner("Consulting AI..."):
                            seo_text_json, seo_err = generate_seo_tags_post_gen(api_key, product_url_input)
                            if seo_text_json:
                                seo_data = parse_json_response(seo_text_json)
                                if seo_data:
                                    with st.expander("✅ Results", expanded=True):
                                        st.write("**File Name:**"); st.code(seo_data.get('file_name', 'N/A'), language="text")
                                        st.write("**Alt Tag:**"); st.code(seo_data.get('alt_tag', 'N/A'), language="text")
                                else: st.code(seo_text_json)
                            else: st.error(seo_err)
        else: st.warning("Library empty.")

# === TAB 2: BULK SEO TAGS ===
with tab2:
    st.header("🏷️ Generate SEO Tags")
    bc1, bc2 = st.columns([1, 1.5])
    with bc1:
        bulk_files = st.file_uploader("Choose images", accept_multiple_files=True, key="bulk_seo_upload")
        bulk_images = [Image.open(f) for f in bulk_files] if bulk_files else []
        if bulk_images: st.success(f"{len(bulk_images)} images selected.")
        
    with bc2:
        bulk_url = st.text_input("Product URL:", key="bulk_seo_url")
        run_bulk_btn = st.button("🚀 Run Batch", type="primary", disabled=(not bulk_images))
        
    if run_bulk_btn:
        if not api_key: st.error("No API Key")
        elif not bulk_url: st.error("No URL")
        else:
            pbar = st.progress(0); res = st.container()
            for i, img_pil in enumerate(bulk_images):
                with st.spinner(f"Processing {i+1}..."):
                    json_txt, err = generate_seo_for_existing_image(api_key, img_pil, bulk_url)
                    pbar.progress((i + 1) / len(bulk_images))
                    with res:
                        rc1, rc2 = st.columns([1, 4])
                        rc1.image(img_pil, width=80)
                        if json_txt:
                            d = parse_json_response(json_txt)
                            if d:
                                with rc2.expander(f"✅ #{i+1}", expanded=True):
                                    st.code(d.get('file_name'), language="text")
                                    st.code(d.get('alt_tag'), language="text")
                            else: rc2.code(json_txt)
                        else: rc2.error(err)
            st.success("Done!")

# === TAB 3: LIBRARY MANAGER ===
with tab3:
    st.subheader("🛠️ Prompt Library")
    target = st.session_state.edit_target
    title = f"Edit: {target['name']}" if target else "Add New"
    with st.form("style_form"):
        st.write(f"**{title}**")
        c1, c2 = st.columns(2)
        n = c1.text_input("Name", value=target['name'] if target else "")
        c = c2.text_input("Category", value=target['category'] if target else "")
        t = st.text_area("Template", value=target['template'] if target else "")
        v = st.text_input("Vars", value=target['variables'] if target else "")
        u = st.text_input("Sample URL", value=target['sample_url'] if target else "")
        
        cols = st.columns([1, 4])
        if cols[0].form_submit_button("💾 Save"):
            new = {"id": target['id'] if target else str(len(st.session_state.library)+1000), "name": n, "category": c, "template": t, "variables": v, "sample_url": u}
            if target:
                for idx, item in enumerate(st.session_state.library):
                    if item['id'] == target['id']: st.session_state.library[idx] = new; break
            else: st.session_state.library.append(new)
            save_prompts(st.session_state.library)
            st.session_state.edit_target = None; st.rerun()
            
        if target and cols[1].form_submit_button("❌ Cancel"):
            st.session_state.edit_target = None; st.rerun()

    st.divider()
    for i, p in enumerate(st.session_state.library):
        c1, c2, c3, c4 = st.columns([1, 4, 1, 1])
        if p.get("sample_url"): c1.image(p["sample_url"], width=50)
        c2.write(f"**{p.get('name')}**")
        if c3.button("✏️", key=f"e{i}"): st.session_state.edit_target = p; st.rerun()
        if c4.button("🗑️", key=f"d{i}"): st.session_state.library.pop(i); save_prompts(st.session_state.library); st.rerun()

# === TAB 4: ABOUT MODELS ===
with tab4:
    st.header("Check Models")
    if st.button("📡 Scan Models"):
        if not api_key: st.error("No API Key")
        else:
            with st.spinner("Scanning..."):
                m = list_available_models(api_key)
                if m:
                    gem = [x for x in m if "gemini" in x['name']]
                    st.success(f"Found {len(gem)} Gemini models")
                    st.dataframe(pd.DataFrame(gem)[['name', 'version', 'displayName']], use_container_width=True)
                else: st.error("Failed to fetch models.")
