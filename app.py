import streamlit as st
import json
import requests
import base64
from io import BytesIO
from PIL import Image
import time
import re

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Jewelry AI Studio")

# Model IDs
# แนะนำให้ใช้ gemini-1.5-flash เพื่อความรวดเร็วและเสถียร
MODEL_IMAGE_GEN = "models/gemini-3-pro-image-preview" 
MODEL_TEXT_SEO = "models/gemini-3-pro-preview"

# --- HELPER: ULTRA CLEANER (ตัวแก้ปัญหาหลัก) ---
def force_clean(value):
    """
    ฟังก์ชันล้างค่าตัวแปร: ลบช่องว่าง, ฟันหนู, และอักขระพิเศษที่ทำให้ URL พัง
    """
    if value is None: return ""
    # แปลงเป็น String -> ตัดช่องว่างหน้าหลัง -> ลบ " และ ' ออก
    return str(value).strip().replace(" ", "").replace('"', "").replace("'", "")

# --- HELPER: CLEAN FILENAME ---
def clean_filename(name):
    if not name: return "N/A"
    # ล้างชื่อไฟล์ให้ปลอดภัย
    clean = re.sub(r'[^a-zA-Z0-9\-\_\.]', '', str(name))
    return clean.rsplit('.', 1)[0]

# --- PROMPTS ---
SEO_PROMPT_POST_GEN = """
You are an SEO specialist. Write SEO-optimized image file name and alt tags in English based on this url: {product_url}.
IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure: {"file_name": "...", "alt_tag": "..."}
"""

SEO_PROMPT_BULK_EXISTING = """
You are an SEO specialist. Write SEO-optimized image file name and alt tags in English based on this url: {product_url}.
IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure: {"file_name": "...", "alt_tag": "..."}
"""

SEO_PRODUCT_WRITER_PROMPT = """
You are an SEO product content writer for Shopify.
Input Data: {raw_input}
Structure: H1, Opening, Body, Specs (Dimension/Weight), FAQ.
Tone: Human-like.

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
    { "file_name": "silver-medusa-ring-mens.jpg", "alt_tag": "Silver Medusa Ring detailed view" },
    { "file_name": "medusa-ring-side-view.jpg", "alt_tag": "Side view of handcrafted Medusa ring" },
    { "file_name": "medusa-ring-on-finger.jpg", "alt_tag": "Model wearing silver Medusa ring" }
  ]
}
"""

# Default Data (Fallback)
DEFAULT_PROMPTS = [
    {
        "id": "default_p1", "name": "Luxury Hand (Ring)", "category": "Ring",
        "template": "A realistic close-up of a female hand model wearing a ring with {face_size} face size, soft studio lighting, elegant jewelry photography.",
        "variables": "face_size",
        "sample_url": "[https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Ring_render.jpg/320px-Ring_render.jpg](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Ring_render.jpg/320px-Ring_render.jpg)"
    }
]

# --- 2. DATABASE FUNCTIONS (FAIL-SAFE MODE) ---
def get_prompts_safe():
    """
    ฟังก์ชันดึงข้อมูลที่ปลอดภัยจากการเว้นวรรคและฟันหนูใน Secrets
    """
    try:
        # ดึงค่าและส่งเข้าเครื่องล้าง force_clean ทันที
        API_KEY = force_clean(st.secrets.get("JSONBIN_API_KEY", ""))
        BIN_ID = force_clean(st.secrets.get("JSONBIN_BIN_ID", ""))

        if not API_KEY or not BIN_ID:
            return DEFAULT_PROMPTS, "Missing Keys in Secrets"

        # สร้าง URL (Clean URL)
        url = f"[https://api.jsonbin.io/v3/b/](https://api.jsonbin.io/v3/b/){BIN_ID}/latest"
        headers = {"X-Master-Key": API_KEY, "X-Bin-Meta": "false"}
        
        # ใส่ Timeout ป้องกันค้าง
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list): return data, None
            elif isinstance(data, dict) and "record" in data: return data["record"], None
            return DEFAULT_PROMPTS, "Unknown JSON format"
        else:
            return DEFAULT_PROMPTS, f"HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        return DEFAULT_PROMPTS, str(e)

def save_prompts_safe(data):
    try:
        API_KEY = force_clean(st.secrets.get("JSONBIN_API_KEY", ""))
        BIN_ID = force_clean(st.secrets.get("JSONBIN_BIN_ID", ""))
        
        if not API_KEY or not BIN_ID:
            st.error("No Database Credentials.")
            return

        url = f"[https://api.jsonbin.io/v3/b/](https://api.jsonbin.io/v3/b/){BIN_ID}"
        headers = {"Content-Type": "application/json", "X-Master-Key": API_KEY}
        
        res = requests.put(url, json=data, headers=headers, timeout=10)
        if res.status_code != 200:
            st.error(f"Save Failed: {res.text}")
            
    except Exception as e:
        st.error(f"Save Error: {e}")

# --- 3. HELPER FUNCTIONS ---
def img_to_base64(img):
    buf = BytesIO()
    if img.mode == 'RGBA': img = img.convert('RGB')
    img.thumbnail((800, 800)) 
    img.save(buf, format="JPEG", quality=85)
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
        clean_url = str(url).strip()
        if clean_url.startswith("http"): st.image(clean_url, width=width)
    except: pass

# --- AI FUNCTIONS ---
def generate_image(api_key, image_list, prompt):
    key = force_clean(api_key)
    # URL ต้องชิดกัน ห้ามมีเว้นวรรค
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_IMAGE_GEN}:generateContent?key={key}"
    
    parts = [{"text": f"Instruction: {prompt}"}]
    for img in image_list: parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    
    for attempt in range(3):
        try:
            res = requests.post(url, json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.3}}, headers={"Content-Type": "application/json"}, timeout=30)
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                if "inline_data" in content: return base64.b64decode(content["inline_data"]["data"]), None
                if "inlineData" in content: return base64.b64decode(content["inlineData"]["data"]), None
                return None, "No image returned."
            elif res.status_code == 503:
                time.sleep((attempt + 1) * 3)
                continue
            else: return None, f"API Error: {res.text}"
        except Exception as e:
            time.sleep(2)
            if attempt == 2: return None, str(e)
    return None, "Failed: Overloaded"

def generate_seo_tags_post_gen(api_key, product_url):
    key = force_clean(api_key)
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_TEXT_SEO}:generateContent?key={key}"
    prompt = SEO_PROMPT_POST_GEN.replace("{product_url}", product_url)
    
    for attempt in range(3):
        try:
            res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={"Content-Type": "application/json"}, timeout=20)
            if res.status_code == 200:
                return res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text"), None
            elif res.status_code == 503: 
                time.sleep(2); continue
            else: return None, f"Error {res.status_code}"
        except Exception as e: time.sleep(1)
    return None, "Failed"

def generate_seo_for_existing_image(api_key, img_pil, product_url):
    key = force_clean(api_key)
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_TEXT_SEO}:generateContent?key={key}"
    prompt = SEO_PROMPT_BULK_EXISTING.replace("{product_url}", product_url)
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}}]}]}
    
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=20)
            if res.status_code == 200:
                return res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text"), None
            elif res.status_code == 503: 
                time.sleep(2); continue
            else: return None, f"Error {res.status_code}"
        except Exception as e: time.sleep(1)
    return None, "Failed"

def generate_full_product_content(api_key, img_pil_list, raw_input):
    key = force_clean(api_key)
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_TEXT_SEO}:generateContent?key={key}"
    prompt = SEO_PRODUCT_WRITER_PROMPT.replace("{raw_input}", raw_input)
    
    parts = [{"text": prompt}]
    if img_pil_list:
        for img in img_pil_list:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}}
    
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                return content.get("text"), None
            elif res.status_code == 503: 
                time.sleep(3); continue
            else: return None, f"Error {res.status_code}"
        except Exception as e: time.sleep(1)
    return None, "Failed"

def list_available_models(api_key):
    # ใช้ force_clean ล้าง key ก่อนเสมอ
    key = force_clean(api_key)
    # URL ห้ามมีเว้นวรรค
    url = f"[https://generativelanguage.googleapis.com/v1beta/models?key=](https://generativelanguage.googleapis.com/v1beta/models?key=){key}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.json().get("models", []), None
        else:
            return None, f"Error: {res.text}"
    except Exception as e: return None, str(e)

# --- 4. UI LOGIC ---
if "db_error_msg" not in st.session_state: st.session_state.db_error_msg = None

# *** STARTUP LOGIC (SAFE) ***
if "library" not in st.session_state:
    data, error = get_prompts_safe()
    st.session_state.library = data
    st.session_state.db_error_msg = error

if "edit_target" not in st.session_state: st.session_state.edit_target = None
if "image_generated_success" not in st.session_state: st.session_state.image_generated_success = False
if "current_generated_image" not in st.session_state: st.session_state.current_generated_image = None

with st.sidebar:
    st.title("💎 Config")
    
    # Key Management (Clean Key ทันทีที่รับมา)
    secret_key = force_clean(st.secrets.get("GEMINI_API_KEY", ""))
    
    if secret_key:
        api_key = secret_key
        st.success("API Key Ready")
    else:
        api_key_input = st.text_input("Gemini API Key", type="password")
        api_key = force_clean(api_key_input)
    
    st.divider()
    
    # DB Status
    db_key_exists = "JSONBIN_API_KEY" in st.secrets
    
    if db_key_exists:
        if st.session_state.db_error_msg:
            st.error("DB Connection Failed")
            with st.expander("Details"):
                st.write(st.session_state.db_error_msg)
                st.caption("Using Default Data")
        else:
            st.caption(f"✅ DB Connected ({len(st.session_state.library)} items)")
            
        if st.button("🔄 Reload DB"):
            data, error = get_prompts_safe()
            st.session_state.library = data
            st.session_state.db_error_msg = error
            st.rerun()
    else:
        st.warning("⚠️ Local Mode (No Secrets)")

st.title("Jewelry AI Studio")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["✨ Gen Image", "🏷️ Bulk SEO", "📝 Writer", "📚 Library", "ℹ️ Models"])

# === TAB 1: GENERATE IMAGE ===
with tab1:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("1. Upload Reference")
        files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"], key="gen_upload")
        images_to_send = [Image.open(f) for f in files] if files else []
        if images_to_send:
            st.caption(f"Selected {len(images_to_send)} images")
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
            
            if st.button("🚀 GENERATE", type="primary", use_container_width=True):
                if not api_key or not images_to_send:
                    st.error("Check Key & Images")
                else:
                    with st.spinner(f"Generating..."):
                        d, e = generate_image(api_key, images_to_send, prompt_edit)
                        if d:
                            st.session_state.current_generated_image = d
                            st.session_state.image_generated_success = True
                            st.rerun()
                        else: st.error(e)

            if st.session_state.image_generated_success and st.session_state.current_generated_image:
                st.divider()
                st.subheader("Result")
                st.image(st.session_state.current_generated_image, use_column_width=True)
                st.download_button("Download", st.session_state.current_generated_image, "gen.jpg", "image/jpeg", type="primary")
                
                st.subheader("Post-Gen SEO")
                url_input = st.text_input("Product URL:", key="post_gen_url")
                if st.button("✨ Gen Tags"):
                    if not url_input: st.warning("Enter URL")
                    else:
                        with st.spinner("Thinking..."):
                            seo_json, err = generate_seo_tags_post_gen(api_key, url_input)
                            if seo_json:
                                data = parse_json_response(seo_json)
                                if data:
                                    with st.expander("Results", expanded=True):
                                        st.write("**File Name:**"); st.code(clean_filename(data.get('file_name')), language="text")
                                        st.write("**Alt Tag:**"); st.code(data.get('alt_tag'), language="text")
                                else: st.code(seo_json)
                            else: st.error(err)

# === TAB 2: BULK SEO ===
with tab2:
    st.header("Bulk SEO Tags")
    bc1, bc2 = st.columns([1, 1.5])
    with bc1:
        files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"], key="bulk_upload")
        imgs = [Image.open(f) for f in files] if files else []
        if imgs: st.success(f"{len(imgs)} selected")
            
    with bc2:
        url = st.text_input("Product URL:", key="bulk_url")
        run_btn = st.button("🚀 Run Batch", type="primary", disabled=(not imgs))

    if run_btn:
        if not api_key or not url: st.error("Check Key & URL")
        else:
            pbar = st.progress(0); res_area = st.container()
            for i, img in enumerate(imgs):
                with st.spinner(f"#{i+1}..."):
                    json_txt, err = generate_seo_for_existing_image(api_key, img, url)
                    pbar.progress((i+1)/len(imgs))
                    with res_area:
                        c1, c2 = st.columns([1, 4])
                        c1.image(img, width=80)
                        if json_txt:
                            d = parse_json_response(json_txt)
                            if d:
                                with c2.expander(f"✅ #{i+1}", expanded=True):
                                    st.write(f"File: `{clean_filename(d.get('file_name'))}`")
                                    st.write(f"Alt: `{d.get('alt_tag')}`")
                            else: c2.code(json_txt)
                        else: c2.error(err)
            st.success("Done!")

# === TAB 3: WRITER ===
with tab3:
    st.header("Product Writer")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        files = st.file_uploader("Images (Optional)", type=["jpg", "png"], accept_multiple_files=True, key="w_img")
        writer_imgs = [Image.open(f) for f in files] if files else []
        raw = st.text_area("Paste Details:", height=300, key="raw_in")
        btn = st.button("🚀 Generate", type="primary")
    with c2:
        if btn:
            if not api_key or not raw: st.error("Missing Info")
            else:
                with st.spinner("Writing..."):
                    json_txt, err = generate_full_product_content(api_key, writer_imgs, raw)
                    if json_txt:
                        d = parse_json_response(json_txt)
                        if d:
                            st.write("Slug:"); st.code(d.get('url_slug'))
                            st.write("Title:"); st.code(d.get('meta_title'))
                            st.write("Desc:"); st.code(d.get('meta_description'))
                            with st.expander("HTML Content"): st.code(d.get('html_content'), language="html")
                            st.markdown(d.get('html_content', ''), unsafe_allow_html=True)
                            
                            st.subheader("Image SEO")
                            img_tags = d.get('image_seo', [])
                            for i, item in enumerate(img_tags):
                                st.write(f"**Img #{i+1}:** `{clean_filename(item.get('file_name', ''))}`")
                                st.caption(item.get('alt_tag', ''))
                        else: st.code(json_txt)
                    else: st.error(err)

# === TAB 4: LIBRARY ===
with tab4:
    st.subheader("Library Manager")
    target = st.session_state.edit_target
    title = f"Edit: {target['name']}" if target else "Add New"
    with st.form("lib_form"):
        st.write(f"**{title}**")
        c1, c2 = st.columns(2)
        n = c1.text_input("Name", value=target['name'] if target else "")
        c = c2.text_input("Category", value=target['category'] if target else "")
        t = st.text_area("Template", value=target['template'] if target else "")
        v = st.text_input("Vars", value=target['variables'] if target else "")
        u = st.text_input("Img URL", value=target['sample_url'] if target else "")
        cols = st.columns([1, 4])
        save = cols[0].form_submit_button("💾 Save")
        if target and cols[1].form_submit_button("❌ Cancel"):
            st.session_state.edit_target = None; st.rerun()
        if save:
            new = {"id": target['id'] if target else str(len(st.session_state.library)+1000), "name": n, "category": c, "template": t, "variables": v, "sample_url": u}
            if target:
                for i, item in enumerate(st.session_state.library):
                    if item['id'] == target['id']: st.session_state.library[i] = new; break
            else: st.session_state.library.append(new)
            save_prompts_safe(st.session_state.library)
            st.session_state.edit_target = None
            st.rerun()
            
    st.divider()
    for i, p in enumerate(st.session_state.library):
        c1, c2, c3, c4 = st.columns([1, 4, 1, 1])
        if p.get("sample_url"): safe_st_image(p["sample_url"], width=50)
        c2.write(f"**{p.get('name')}**")
        if c3.button("✏️", key=f"e{i}"): st.session_state.edit_target = p; st.rerun()
        if c4.button("🗑️", key=f"d{i}"): st.session_state.library.pop(i); save_prompts_safe(st.session_state.library); st.rerun()

# === TAB 5: MODELS ===
with tab5:
    st.header("Check Models")
    if st.button("Scan Models"):
        if not api_key: st.error("No Key")
        else:
            with st.spinner("Scanning..."):
                m, err = list_available_models(api_key)
                if m:
                    gem = [{"ID": x['name']} for x in m if "gemini" in x['name']]
                    st.success(f"Found {len(gem)} Gemini models")
                    st.dataframe(gem)
                else: 
                    st.error("Scan failed")
                    if err: st.error(err)

