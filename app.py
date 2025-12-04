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

# Model IDs
MODEL_IMAGE_GEN = "models/gemini-3-pro-image-preview"
MODEL_TEXT_SEO = "models/gemini-3-pro-preview"

# --- PROMPTS ---
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

SEO_PROMPT_BULK_EXISTING = """
คุณคือ SEO specialist ที่มีประสบการณ์ 15-20 ปี ช่วยเขียน SEO-optimized image file name with image alt tags เป็นภาษาอังกฤษ สำหรับสินค้าของฉันตามแต่ละรูปที่แนบมาให้ โดยมีรายละเอียดของสินค้าตาม url นี้ {product_url} เพื่อให้ได้ติดอันดับที่ดีบน organic search engine โดยกลุ่มลูกค้าเป็นผู้สนใจสินค้าชนิดนี้

IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure:
{
  "file_name": "your-optimized-filename.jpg",
  "alt_tag": "Your optimized descriptive alt tag"
}
"""

SEO_PRODUCT_WRITER_PROMPT = """
คุณมีหน้าที่เป็นผู้เชี่ยวชาญ SEO specialist product content writer ผู้มีประสบการ์ 15-20 ปี ช่วยเขียน SEO-Optimized product description เป็นภาษาอังกฤษสำหรับร้าน e-commerce ของฉันที่สร้างโดย Shopify

**INPUT DATA (ข้อมูลสินค้า):**
{raw_input}

**คำสั่งการเขียน:**
จากข้อมูล Input Data ด้านบน ให้คุณวิเคราะห์หา URL, Primary Keyword, Secondary Keywords, Category และรายละเอียดสินค้า แล้วเขียนบทความตามโครงสร้างนี้:

1. **Product Title (H1):** น่าสนใจและมีคีย์เวิร์ด
2. **Opening Paragraph:** บอก Google และผู้ใช้ให้ชัดเจนว่าหน้านี้คืออะไร (เน้น Primary Keyword + Semantic 1-2 คำ)
3. **Body Content:** เล่าเรื่องราว, ดีไซน์, สัญลักษณ์ (กระจาย Semantic Keywords อย่างเป็นธรรมชาติ)
4. **Specifications:** ใช้ Bullet Points (<ul><li>) เน้นวัสดุ และ **ต้องระบุ Dimension (ขนาด) และ Weight (น้ำหนัก)** ถ้ามีข้อมูล
5. **FAQ Section:** ตอบข้อสงสัย (ใช้ Long-tail keywords)

Tone: Human-written style, Fact-driven, อ่านง่าย, ดึงดูดลูกค้า, ต้องผ่าน AI Content detector (Undetectable.ai)

**IMPORTANT OUTPUT FORMAT:**
You MUST return the result in **RAW JSON** format ONLY. Do not include markdown backticks (```json).
The JSON structure must be exactly like this:
{
  "url_slug": "url-slug-example",
  "meta_title": "Meta Title Example (Max 60 chars)",
  "meta_description": "Meta Description Example (Max 160 chars)",
  "product_title_h1": "Product Title Example",
  "html_content": "<p>Your full HTML product description here...</p>",
  "image_seo": [
    { "file_name": "img1.jpg", "alt_tag": "alt tag 1" },
    { "file_name": "img2.jpg", "alt_tag": "alt tag 2" },
    { "file_name": "img3.jpg", "alt_tag": "alt tag 3" },
    { "file_name": "img4.jpg", "alt_tag": "alt tag 4" },
    { "file_name": "img5.jpg", "alt_tag": "alt tag 5" },
    { "file_name": "img6.jpg", "alt_tag": "alt tag 6" },
    { "file_name": "img7.jpg", "alt_tag": "alt tag 7" },
    { "file_name": "img8.jpg", "alt_tag": "alt tag 8" }
  ]
}
"""

# Default Data
DEFAULT_PROMPTS = [
    {
        "id": "p1", "name": "Luxury Hand (Ring)", "category": "Ring",
        "template": "A realistic close-up of a female hand model wearing a ring with {face_size} face size, soft studio lighting, elegant jewelry photography.",
        "variables": "face_size",
        "sample_url": "[https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Ring_render.jpg/320px-Ring_render.jpg](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Ring_render.jpg/320px-Ring_render.jpg)"
    }
]

# --- 2. CLOUD DATABASE FUNCTIONS ---
def clean_key(key):
    """ฟังก์ชันทำความสะอาด API Key (ลบช่องว่าง, ลบ quotes)"""
    if not key: return ""
    return str(key).strip().strip('"').strip("'")

def get_prompts():
    if "JSONBIN_API_KEY" in st.secrets and "JSONBIN_BIN_ID" in st.secrets:
        try:
            # Clean Keys ก่อนใช้เสมอ
            API_KEY = clean_key(st.secrets["JSONBIN_API_KEY"])
            BIN_ID = clean_key(st.secrets["JSONBIN_BIN_ID"])
            
            url = f"[https://api.jsonbin.io/v3/b/](https://api.jsonbin.io/v3/b/){BIN_ID}/latest"
            headers = {"X-Master-Key": API_KEY, "X-Bin-Meta": "false"}
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list): return data
                elif isinstance(data, dict) and "record" in data: return data["record"]
                else: return DEFAULT_PROMPTS
            else:
                return DEFAULT_PROMPTS
        except: return DEFAULT_PROMPTS
    else: return DEFAULT_PROMPTS

def save_prompts(data):
    if "JSONBIN_API_KEY" in st.secrets and "JSONBIN_BIN_ID" in st.secrets:
        try:
            API_KEY = clean_key(st.secrets["JSONBIN_API_KEY"])
            BIN_ID = clean_key(st.secrets["JSONBIN_BIN_ID"])
            url = f"[https://api.jsonbin.io/v3/b/](https://api.jsonbin.io/v3/b/){BIN_ID}"
            headers = {"Content-Type": "application/json", "X-Master-Key": API_KEY}
            requests.put(url, json=data, headers=headers)
        except Exception as e: st.error(f"Save Error: {e}")
    else: st.warning("No DB Credentials.")

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
    except: return None

def safe_st_image(url, width=None):
    if not url: return
    try:
        if url.startswith("http"): st.image(url, width=width)
    except: pass

# --- AI FUNCTIONS ---
def generate_image(api_key, image_list, prompt):
    api_key = clean_key(api_key) # Clean Key Here
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_IMAGE_GEN}:generateContent?key={api_key}"
    parts = [{"text": f"Instruction: {prompt} \nConstraint: Keep the jewelry products in the input images EXACTLY as they are. Analyze all images to understand the 3D structure. Generate a realistic model wearing it."}]
    for img in image_list:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    try:
        res = requests.post(url, json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.3}}, headers={"Content-Type": "application/json"})
        if res.status_code != 200: return None, f"API Error: {res.text}"
        content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
        if "inline_data" in content: return base64.b64decode(content["inline_data"]["data"]), None
        if "inlineData" in content: return base64.b64decode(content["inlineData"]["data"]), None
        return None, "No image returned."
    except Exception as e: return None, str(e)

def generate_seo_tags_post_gen(api_key, product_url):
    api_key = clean_key(api_key) # Clean Key Here
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_TEXT_SEO}:generateContent?key={api_key}"
    final_seo_prompt = SEO_PROMPT_POST_GEN.replace("{product_url}", product_url)
    payload = {"contents": [{"parts": [{"text": final_seo_prompt}]}], "generationConfig": {"temperature": 0.5, "responseMimeType": "application/json"}}
    
    for _ in range(3):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                return content.get("text"), None
            elif res.status_code == 503: time.sleep(2); continue
            else: return None, res.text
        except: time.sleep(2)
    return None, "Failed after retries"

def generate_seo_for_existing_image(api_key, img_pil, product_url):
    api_key = clean_key(api_key) # Clean Key Here
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_TEXT_SEO}:generateContent?key={api_key}"
    final_prompt = SEO_PROMPT_BULK_EXISTING.replace("{product_url}", product_url)
    payload = {"contents": [{"parts": [{"text": final_prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}}]}], "generationConfig": {"temperature": 0.5, "maxOutputTokens": 2048, "responseMimeType": "application/json"}}
    
    for _ in range(3):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                return content.get("text"), None
            elif res.status_code == 503: time.sleep(2); continue
            else: return None, res.text
        except: time.sleep(2)
    return None, "Failed after retries"

def generate_full_product_content(api_key, img_pil, raw_input):
    api_key = clean_key(api_key) # Clean Key Here
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_TEXT_SEO}:generateContent?key={api_key}"
    final_prompt = SEO_PRODUCT_WRITER_PROMPT.replace("{raw_input}", raw_input)
    parts = [{"text": final_prompt}]
    if img_pil: parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}})
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}}
    
    for _ in range(3):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                return content.get("text"), None
            elif res.status_code == 503: time.sleep(2); continue
            else: return None, res.text
        except: time.sleep(2)
    return None, "Failed after retries"

def list_available_models(api_key):
    api_key = clean_key(api_key) # Clean Key Here
    url = f"[https://generativelanguage.googleapis.com/v1beta/models?key=](https://generativelanguage.googleapis.com/v1beta/models?key=){api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("models", []), None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e: return None, str(e)

# --- 4. UI LOGIC ---
if "library" not in st.session_state: st.session_state.library = get_prompts()
if "edit_target" not in st.session_state: st.session_state.edit_target = None
if "image_generated_success" not in st.session_state: st.session_state.image_generated_success = False
if "current_generated_image" not in st.session_state: st.session_state.current_generated_image = None

with st.sidebar:
    st.title("⚙️ Config")
    try:
        # Load and CLEAN the key from secrets
        api_key = clean_key(st.secrets["GEMINI_API_KEY"])
        st.success("API Key Ready")
    except:
        api_key_input = st.text_input("Gemini API Key", type="password")
        api_key = clean_key(api_key_input) # Clean input key
    
    if "JSONBIN_API_KEY" in st.secrets:
        st.caption(f"✅ DB Connected ({len(st.session_state.library)} items)")
    else:
        st.warning("⚠️ Local Mode")

st.title("💎 Jewelry AI Studio")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["✨ Gen Image", "🏷️ Bulk SEO", "📝 Writer", "📚 Library", "ℹ️ Models"])

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
            if sel_style.get("sample_url"): safe_st_image(sel_style["sample_url"], width=100)
            
            vars_list = [v.strip() for v in sel_style.get('variables', '').split(",") if v.strip()]
            user_vals = {v: st.text_input(v, placeholder="e.g. Gold") for v in vars_list}
            
            final_prompt = sel_style.get('template', '')
            for k, v in user_vals.items(): final_prompt = final_prompt.replace(f"{{{k}}}", v)
            
            st.write("✏️ **Edit Prompt:**")
            prompt_edit = st.text_area("Instruction", value=final_prompt, height=100)
            
            if st.button("🚀 GENERATE IMAGE", type="primary", use_container_width=True):
                if not api_key or not images_to_send:
                    st.error("Check Key & Images")
                    st.session_state.image_generated_success = False
                    st.session_state.current_generated_image = None
                else:
                    with st.spinner(f"Generating Image ({MODEL_IMAGE_GEN})..."):
                        d, e = generate_image(api_key, images_to_send, prompt_edit)
                        if d:
                            st.session_state.current_generated_image = d
                            st.session_state.image_generated_success = True
                            st.rerun()
                        else: st.error(e)

            if st.session_state.image_generated_success and st.session_state.current_generated_image:
                st.divider()
                st.subheader("🎉 Result")
                st.image(st.session_state.current_generated_image, use_column_width=True)
                st.download_button("Download", st.session_state.current_generated_image, "gen.jpg", "image/jpeg", type="primary")
                
                st.divider()
                st.subheader("🌍 Post-Generation SEO")
                url_input = st.text_input("Product URL:", key="post_gen_url")
                if st.button("✨ Gen Tags"):
                    if not url_input: st.warning("Enter URL")
                    else:
                        with st.spinner("Thinking..."):
                            seo_json, err = generate_seo_tags_post_gen(api_key, url_input)
                            if seo_json:
                                data = parse_json_response(seo_json)
                                if data:
                                    with st.expander("✅ Results", expanded=True):
                                        st.write("**File Name:**"); st.code(data.get('file_name'), language="text")
                                        st.write("**Alt Tag:**"); st.code(data.get('alt_tag'), language="text")
                                else: st.code(seo_json)
                            else: st.error(err)
        else: st.warning("Library empty.")

# === TAB 2: BULK SEO ===
with tab2:
    st.header("🏷️ Bulk SEO Tags")
    bc1, bc2 = st.columns([1, 1.5])
    with bc1:
        files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"], key="bulk_upload")
        imgs = [Image.open(f) for f in files] if files else []
        if imgs:
            st.success(f"{len(imgs)} selected")
            cols = st.columns(4)
            for i, img in enumerate(imgs): cols[i%4].image(img, use_column_width=True, caption=f"#{i+1}")
    with bc2:
        url = st.text_input("Product URL:", key="bulk_url")
        run_btn = st.button("🚀 Run Batch", type="primary", disabled=(not imgs))

    if run_btn:
        if not api_key or not url: st.error("Check Key & URL")
        else:
            pbar = st.progress(0); res_area = st.container()
            for i, img in enumerate(imgs):
                with st.spinner(f"Processing #{i+1}..."):
                    json_txt, err = generate_seo_for_existing_image(api_key, img, url)
                    pbar.progress((i+1)/len(imgs))
                    with res_area:
                        c1, c2 = st.columns([1, 4])
                        c1.image(img, width=80, caption=f"#{i+1}")
                        if json_txt:
                            d = parse_json_response(json_txt)
                            if d:
                                with c2.expander(f"✅ #{i+1} Tags", expanded=True):
                                    st.write("**File:**"); st.code(d.get('file_name'), language="text")
                                    st.write("**Alt:**"); st.code(d.get('alt_tag'), language="text")
                            else: c2.code(json_txt)
                        else: c2.error(err)
                    time.sleep(0.5)
            st.success("Done!")

# === TAB 3: WRITER ===
with tab3:
    st.header("📝 AI Product Writer")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        f = st.file_uploader("Product Image (Optional)", type=["jpg", "png"], key="w_img")
        img = Image.open(f) if f else None
        if img: st.image(img, width=200)
        raw = st.text_area("Paste Raw Details:", height=300, key="raw_in")
        btn = st.button("🚀 Generate Content", type="primary")
    with c2:
        if btn:
            if not api_key or not raw: st.error("Missing Info")
            else:
                with st.spinner("Writing..."):
                    json_txt, err = generate_full_product_content(api_key, img, raw)
                    if json_txt:
                        d = parse_json_response(json_txt)
                        if d:
                            st.write("🔗 **Slug:**"); st.code(d.get('url_slug'), language="text")
                            st.write("🪪 **Title:**"); st.code(d.get('meta_title'), language="text")
                            st.write("📝 **Desc:**"); st.code(d.get('meta_description'), language="text")
                            st.write("📌 **H1:**"); st.code(d.get('product_title_h1'), language="text")
                            st.write("📄 **HTML:**"); st.code(d.get('html_content'), language="html")
                            with st.expander("Preview"): st.markdown(d.get('html_content', ''), unsafe_allow_html=True)
                            st.divider()
                            for i, item in enumerate(d.get('image_seo', [])):
                                with st.container():
                                    cols = st
