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
st.set_page_config(layout="wide", page_title="Jewelry AI Studio 12/9")

# Model IDs (Updated to Gemini 3 as requested)
MODEL_IMAGE_GEN = "models/gemini-3-pro-image-preview" # แนะนำใช้ Flash 2.0 หรือ 1.5 เพราะเสถียรกว่า 3-preview
MODEL_TEXT_SEO = "models/gemini-3-pro-preview"   # ถ้า 3 error บ่อย ให้ลองเปลี่ยนเป็น gemini-1.5-pro

# --- HELPER: CLEANER ---
def clean_key(value):
    """ล้างค่า Key/ID ให้สะอาด ป้องกันช่องว่างและฟันหนู"""
    if value is None: return ""
    return str(value).strip().replace(" ", "").replace('"', "").replace("'", "").replace("\n", "")

# --- HELPER: SAFE IMAGE LOADER ---
def safe_st_image(url, width=None, caption=None):
    if not url: return
    try:
        clean_url = str(url).strip().replace(" ", "").replace("\n", "")
        if clean_url.startswith("http"):
            st.image(clean_url, width=width, caption=caption)
    except Exception:
        st.warning("⚠️ Image unavailable")

# --- PROMPTS ---
SEO_PROMPT_POST_GEN = """
You are an SEO specialist with 15-20 years of experience. 
Help write SEO-optimized image file name with image alt tags in English for the product image with a model created, having product details according to this url: {product_url}
To rank well on organic search engines by customer groups interested in this type of product.

IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure: {"file_name": "...", "alt_tag": "..."}
"""

SEO_PROMPT_BULK_EXISTING = """
คุณคือ SEO specialist ที่มีประสบการณ์ 15-20 ปี ช่วยเขียน SEO-optimized image file name with image alt tags เป็นภาษาอังกฤษ สำหรับสินค้าของฉันตามแต่ละรูปที่แนบมาให้ {product_url} เพื่อให้ได้ติดอันดับที่ดีบน organic search engine โดยกลุ่มลูกค้าเป็นผู้สนใจสินค้าชนิดนี้
IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure: {"file_name": "...", "alt_tag": "..."}
"""

SEO_PRODUCT_WRITER_PROMPT = """
You are an SEO product content writer for Shopify.
Input Data: {raw_input}
Structure: H1, Opening, Body, Specs (Dimension/Weight), FAQ.
Tone: Human-like.

IMPORTANT OUTPUT FORMAT:
You MUST return the result in RAW JSON format ONLY. Do not include markdown backticks.
The JSON structure must be exactly like this:
{
  "url_slug": "url-slug-example",
  "meta_title": "Meta Title Example (Max 60 chars)",
  "meta_description": "Meta Description Example (Max 160 chars)",
  "product_title_h1": "Product Title Example",
  "html_content": "<p>Your full HTML product description here...</p>",
  "image_seo": [
    { "file_name": "silver-medusa-ring-mens.jpg", "alt_tag": "Silver Medusa Ring detailed view" },
    { "file_name": "medusa-ring-side-view.jpg", "alt_tag": "Side view of handcrafted Medusa ring" }
  ]
}
"""

# Default Data
DEFAULT_PROMPTS = [
    {
        "id": "p1", "name": "Luxury Hand (Ring)", "category": "Ring",
        "template": "A realistic close-up of a female hand model wearing a ring with {face_size} face size, soft studio lighting, elegant jewelry photography.",
        "variables": "face_size",
        "sample_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Ring_render.jpg/320px-Ring_render.jpg"
    }
]

# --- 2. CLOUD DATABASE FUNCTIONS ---
def get_prompts():
    try:
        raw_key = st.secrets.get("JSONBIN_API_KEY", "")
        raw_bin = st.secrets.get("JSONBIN_BIN_ID", "")
        API_KEY = clean_key(raw_key)
        BIN_ID = clean_key(raw_bin)

        if not API_KEY or not BIN_ID: return DEFAULT_PROMPTS

        url = f"https://api.jsonbin.io/v3/b/{BIN_ID}/latest"
        headers = {"X-Master-Key": API_KEY}
        
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json().get("record", DEFAULT_PROMPTS)
        return DEFAULT_PROMPTS
    except:
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
    except: return None

def clean_filename(name):
    if not name: return "N/A"
    clean = re.sub(r'[^a-zA-Z0-9\-\_\.]', '', str(name))
    return clean.rsplit('.', 1)[0]

# --- AI FUNCTIONS ---
def generate_image(api_key, image_list, prompt):
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_IMAGE_GEN}:generateContent?key={key}"
    parts = [{"text": f"Instruction: {prompt} \nConstraint: Keep the jewelry products in the input images EXACTLY as they are. Analyze all images to understand the 3D structure. Generate a realistic model wearing it."}]
    for img in image_list: parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    
    try:
        res = requests.post(url, json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.3}}, headers={"Content-Type": "application/json"})
        if res.status_code != 200: return None, f"API Error {res.status_code}: {res.text}"
        content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
        if "inline_data" in content: return base64.b64decode(content["inline_data"]["data"]), None
        if "inlineData" in content: return base64.b64decode(content["inlineData"]["data"]), None
        if "text" in content: return None, f"Model returned text: {content['text']}"
        return None, "Unknown format"
    except Exception as e: return None, str(e)

def generate_seo_tags_post_gen(api_key, product_url):
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_TEXT_SEO}:generateContent?key={key}"
    prompt = SEO_PROMPT_POST_GEN.replace("{product_url}", product_url)
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.5, "responseMimeType": "application/json"}}
    
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                return content.get("text"), None
            elif res.status_code == 503: time.sleep(2); continue
            else: return None, f"Error {res.status_code}"
        except Exception as e: time.sleep(1)
    return None, "Failed"

def generate_seo_for_existing_image(api_key, img_pil, product_url):
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_TEXT_SEO}:generateContent?key={key}"
    prompt = SEO_PROMPT_BULK_EXISTING.replace("{product_url}", product_url)
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}}]}], "generationConfig": {"temperature": 0.5, "responseMimeType": "application/json"}}
    
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                return content.get("text"), None
            elif res.status_code == 503: time.sleep(2); continue
            else: return None, f"Error {res.status_code}"
        except Exception as e: time.sleep(1)
    return None, "Failed"

def generate_full_product_content(api_key, img_pil_list, raw_input):
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_TEXT_SEO}:generateContent?key={key}"
    prompt = SEO_PRODUCT_WRITER_PROMPT.replace("{raw_input}", raw_input)
    parts = [{"text": prompt}]
    if img_pil_list:
        for img in img_pil_list: parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}}
    
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                return content.get("text"), None
            elif res.status_code == 503: time.sleep(3); continue
            else: return None, f"Error {res.status_code}: {res.text}"
        except Exception as e: time.sleep(1)
    return None, "Failed"

def list_available_models(api_key):
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200: return response.json().get("models", [])
        return None
    except: return None

# --- UI LOGIC ---
if "library" not in st.session_state: st.session_state.library = get_prompts()
if "edit_target" not in st.session_state: st.session_state.edit_target = None
if "image_generated_success" not in st.session_state: st.session_state.image_generated_success = False
if "current_generated_image" not in st.session_state: st.session_state.current_generated_image = None

with st.sidebar:
    st.title("⚙️ Config")
    
    # --- AUTO-LOAD SECRETS LOGIC ---
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ API Key Loaded (GEMINI)")
    elif "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ API Key Loaded (GOOGLE)")
    else:
        api_key = st.text_input("Gemini API Key", type="password")
    
    api_key = clean_key(api_key)
    
    if "JSONBIN_API_KEY" in st.secrets:
        st.caption("✅ Database Connected")
    else:
        st.warning("⚠️ Local Mode (DB Not Connected)")

st.title("💎 Jewelry AI Studio")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["✨ Gen Image", "🏷️ Bulk SEO", "📝 Writer", "📚 Library", "ℹ️ Models"])

# === TAB 1: GEN IMAGE ===
with tab1:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("1. Upload Reference")
        files = st.file_uploader("Upload", accept_multiple_files=True, type=["jpg","png"], key="gen_up")
        images_to_send = [Image.open(f) for f in files] if files else []
        if images_to_send:
            cols = st.columns(4)
            for i, img in enumerate(images_to_send): cols[i%4].image(img, use_column_width=True)

    with c2:
        st.subheader("2. Settings")
        lib = st.session_state.library
        cats = list(set(p.get('category','Other') for p in lib)) if lib else []
        sel_cat = st.selectbox("Category", cats) if cats else None
        
        filtered = [p for p in lib if p.get('category') == sel_cat]
        if filtered:
            sel_style = st.selectbox("Style", filtered, format_func=lambda x: x.get('name','Unknown'))
            if sel_style.get("sample_url"): safe_st_image(sel_style["sample_url"], width=100)
            
            vars_list = [v.strip() for v in sel_style.get('variables','').split(",") if v.strip()]
            user_vals = {v: st.text_input(v) for v in vars_list}
            
            final_prompt = sel_style.get('template','')
            for k, v in user_vals.items(): final_prompt = final_prompt.replace(f"{{{k}}}", v)
            
            st.write("✏️ **Edit Prompt:**")
            prompt_edit = st.text_area("Instruction", value=final_prompt, height=100)
            
            if st.button("🚀 GENERATE", type="primary", use_container_width=True):
                if not api_key or not images_to_send: st.error("Check Key & Images")
                else:
                    with st.spinner("Generating..."):
                        d, e = generate_image(api_key, images_to_send, prompt_edit)
                        if d:
                            st.session_state.current_generated_image = d
                            st.session_state.image_generated_success = True
                            st.rerun()
                        else: st.error(e)

            if st.session_state.image_generated_success and st.session_state.current_generated_image:
                st.divider()
                st.image(st.session_state.current_generated_image, use_column_width=True)
                st.download_button("Download", st.session_state.current_generated_image, "gen.jpg", "image/jpeg", type="primary")
                st.divider()
                url_input = st.text_input("Product URL:", key="post_url")
                if st.button("✨ Gen Tags"):
                    if not url_input: st.warning("Enter URL")
                    else:
                        with st.spinner("Thinking..."):
                            txt, err = generate_seo_tags_post_gen(api_key, url_input)
                            if txt:
                                d = parse_json_response(txt)
                                if d:
                                    with st.expander("Results", expanded=True):
                                        st.code(d.get('file_name'), language="text")
                                        st.code(d.get('alt_tag'), language="text")
                                else: st.code(txt)
                            else: st.error(err)

# === TAB 2: BULK SEO (Fixed AttributeError) ===
with tab2:
    st.header("🏷️ Bulk SEO Tags")
    bc1, bc2 = st.columns([1, 1.5])
    with bc1:
        bfiles = st.file_uploader("Upload Images", accept_multiple_files=True, key="bulk_up")
        bimgs = [Image.open(f) for f in bfiles] if bfiles else []
        if bimgs:
            st.success(f"{len(bimgs)} images selected")
            with st.expander("📸 Image Preview", expanded=False):
                cols = st.columns(4)
                for i, img in enumerate(bimgs):
                    cols[i%4].image(img, use_column_width=True, caption=f"Img #{i+1}")

    with bc2:
        burl = st.text_input("Product URL:", key="bulk_url")
        if st.button("🚀 Run Batch", type="primary", disabled=(not bimgs)):
            if not api_key or not burl: st.error("Missing Info")
            else:
                pbar = st.progress(0); res = st.container()
                for i, img in enumerate(bimgs):
                    with st.spinner(f"Processing Image #{i+1}..."):
                        txt, err = generate_seo_for_existing_image(api_key, img, burl)
                        pbar.progress((i+1)/len(bimgs))
                        
                        with res:
                            with st.container():
                                rc1, rc2 = st.columns([1, 3])
                                with rc1:
                                    st.image(img, width=150, caption=f"Img #{i+1}")
                                with rc2:
                                    if txt:
                                        d = parse_json_response(txt)
                                        # --- FIX START: Handle List/Dict Mismatch ---
                                        if isinstance(d, list) and len(d) > 0:
                                            d = d[0] # ถ้าได้มาเป็น List ให้หยิบตัวแรก
                                        
                                        if isinstance(d, dict): # เช็คว่าเป็น Dict ชัวร์ๆ ถึงค่อยใช้ .get
                                            st.write("**✅ Result:**")
                                            st.text_input(f"File Name #{i+1}", value=d.get('file_name', ''), key=f"fn_{i}")
                                            st.text_area(f"Alt Tag #{i+1}", value=d.get('alt_tag', ''), key=f"at_{i}", height=70)
                                        else:
                                            st.warning(f"⚠️ Invalid format returned for image #{i+1}")
                                            st.code(txt)
                                        # --- FIX END ---
                                    else: st.error(err)
                                st.divider()
                st.success("Done!")

# === TAB 3: WRITER ===
with tab3:
    st.header("📝 Product Writer")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        files = st.file_uploader("Images (Optional)", type=["jpg", "png"], accept_multiple_files=True, key="w_img")
        writer_imgs = [Image.open(f) for f in files] if files else []
        
        if writer_imgs:
            with st.expander("📸 Image Preview", expanded=False):
                cols = st.columns(4)
                for i, img in enumerate(writer_imgs):
                    cols[i%4].image(img, use_column_width=True)

        raw = st.text_area("Paste Details:", height=300, key="w_raw")
        btn = st.button("🚀 Generate Content", type="primary")
    with c2:
        if btn:
            if not api_key or not raw: st.error("Missing Info")
            else:
                with st.spinner("Writing..."):
                    json_txt, err = generate_full_product_content(api_key, writer_imgs, raw)
                    if json_txt:
                        d = parse_json_response(json_txt)
                        # Safety check for Writer as well
                        if isinstance(d, list) and len(d) > 0: d = d[0]
                        
                        if isinstance(d, dict):
                            st.subheader("Content Results")
                            st.write("Product Title (H1):"); st.code(d.get('product_title_h1', ''))
                            st.write("Slug Handle:"); st.code(d.get('url_slug', ''))
                            st.write("Meta Title:"); st.code(d.get('meta_title', ''))
                            st.write("Meta Description:"); st.code(d.get('meta_description', ''))
                            with st.expander("HTML Content"): st.code(d.get('html_content', ''), language="html")
                            st.markdown(d.get('html_content', ''), unsafe_allow_html=True)
                            
                            st.divider()
                            st.subheader("🖼️ Image SEO Mapping")
                            
                            img_tags = d.get('image_seo', [])
                            for i, item in enumerate(img_tags):
                                with st.container():
                                    ic1, ic2 = st.columns([1, 3])
                                    with ic1:
                                        if writer_imgs and i < len(writer_imgs):
                                            st.image(writer_imgs[i], width=120, caption=f"Upload #{i+1}")
                                        else:
                                            st.info(f"Image #{i+1}")
                                    
                                    with ic2:
                                        fname = clean_filename(item.get('file_name', 'N/A')) if isinstance(item, dict) else "N/A"
                                        atag = item.get('alt_tag', 'N/A') if isinstance(item, dict) else str(item)
                                        
                                        st.write(f"**File Name:** `{fname}`")
                                        st.write(f"**Alt Tag:** `{atag}`")
                                st.divider()
                        else: st.code(json_txt)
                    else: st.error(err)

# === TAB 4: LIBRARY ===
with tab4:
    st.subheader("🛠️ Library Manager")
    target = st.session_state.edit_target
    title = f"Edit: {target['name']}" if target else "Add New"
    with st.form("lib_form"):
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
        if p.get("sample_url"): 
            with c1: safe_st_image(p["sample_url"], width=50)
            
        c2.write(f"**{p.get('name')}**")
        if c3.button("✏️", key=f"e{i}"): st.session_state.edit_target = p; st.rerun()
        if c4.button("🗑️", key=f"d{i}"): st.session_state.library.pop(i); save_prompts(st.session_state.library); st.rerun()

# === TAB 5: MODELS ===
with tab5:
    if st.button("📡 Scan Models"):
        if not api_key: st.error("No API Key")
        else:
            with st.spinner("Scanning..."):
                m = list_available_models(api_key)
                if m:
                    gem = [x for x in m if "gemini" in x['name']]
                    st.success(f"Found {len(gem)} Gemini models")
                    st.dataframe(pd.DataFrame(gem)[['name','version','displayName']], use_container_width=True)
                else: st.error("Failed to fetch models")





