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
MODEL_IMAGE_GEN = "models/gemini-1.5-flash"
MODEL_TEXT_SEO = "models/gemini-1.5-flash"

# --- HELPER: SUPER CLEANER (ล้างทุกอย่างรวมถึง Newline) ---
def force_clean(value):
    """
    ล้างค่าตัวแปร: ลบช่องว่าง, ฟันหนู, และอักขระซ่อนเร้น (\n, \r)
    """
    if value is None: return ""
    # 1. แปลงเป็น String
    s = str(value)
    # 2. ลบ Newline/Tab ที่มองไม่เห็น
    s = s.replace("\n", "").replace("\r", "").replace("\t", "")
    # 3. ลบ Quote และช่องว่าง
    s = s.strip().replace(" ", "").replace('"', "").replace("'", "")
    return s

def clean_filename(name):
    if not name: return "N/A"
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

DEFAULT_PROMPTS = [
    {
        "id": "default_p1", "name": "Luxury Hand (Ring)", "category": "Ring",
        "template": "A realistic close-up of a female hand model wearing a ring with {face_size} face size, soft studio lighting, elegant jewelry photography.",
        "variables": "face_size",
        "sample_url": "[https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Ring_render.jpg/320px-Ring_render.jpg](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Ring_render.jpg/320px-Ring_render.jpg)"
    }
]

# --- 2. DATABASE FUNCTIONS (WITH DEBUG) ---
def get_prompts_safe(api_key, bin_id):
    try:
        if not api_key or not bin_id:
            return DEFAULT_PROMPTS, "Missing Keys"

        # Construct URL cleanly
        url = f"[https://api.jsonbin.io/v3/b/](https://api.jsonbin.io/v3/b/){bin_id}/latest"
        headers = {"X-Master-Key": api_key, "X-Bin-Meta": "false"}
        
        # Explicit request with timeout
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

def save_prompts_safe(data, api_key, bin_id):
    try:
        if not api_key or not bin_id:
            st.error("No Database Credentials.")
            return

        url = f"[https://api.jsonbin.io/v3/b/](https://api.jsonbin.io/v3/b/){bin_id}"
        headers = {"Content-Type": "application/json", "X-Master-Key": api_key}
        
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
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_IMAGE_GEN}:generateContent?key={api_key}"
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
            elif res.status_code == 503: time.sleep(2); continue
            else: return None, f"API Error: {res.text}"
        except Exception as e: return None, str(e)
    return None, "Overloaded"

def generate_seo_tags_post_gen(api_key, product_url):
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_TEXT_SEO}:generateContent?key={api_key}"
    prompt = SEO_PROMPT_POST_GEN.replace("{product_url}", product_url)
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={"Content-Type": "application/json"}, timeout=20)
        if res.status_code == 200: return res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text"), None
        else: return None, f"Error {res.status_code}"
    except Exception as e: return None, str(e)

def generate_seo_for_existing_image(api_key, img_pil, product_url):
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_TEXT_SEO}:generateContent?key={api_key}"
    prompt = SEO_PROMPT_BULK_EXISTING.replace("{product_url}", product_url)
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}}]}]}
    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=20)
        if res.status_code == 200: return res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text"), None
        else: return None, f"Error {res.status_code}"
    except Exception as e: return None, str(e)

def generate_full_product_content(api_key, img_pil_list, raw_input):
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_TEXT_SEO}:generateContent?key={api_key}"
    prompt = SEO_PRODUCT_WRITER_PROMPT.replace("{raw_input}", raw_input)
    parts = [{"text": prompt}]
    if img_pil_list:
        for img in img_pil_list: parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}}
    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        if res.status_code == 200: return res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text"), None
        else: return None, f"Error {res.status_code}"
    except Exception as e: return None, str(e)

def list_available_models(api_key):
    url = f"[https://generativelanguage.googleapis.com/v1beta/models?key=](https://generativelanguage.googleapis.com/v1beta/models?key=){api_key}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200: return res.json().get("models", []), None
        else: return None, f"Error: {res.text}"
    except Exception as e: return None, str(e)

# --- 4. MAIN UI ---
st.title("Jewelry AI Studio (v2.0 DEBUG)")

# *** SIDEBAR & KEY MANAGEMENT ***
with st.sidebar:
    st.title("💎 Config")
    
    # ดึงค่าจาก Secrets และ Text Input
    # ใช้ force_clean ทันทีเพื่อความปลอดภัย
    secret_gemini = force_clean(st.secrets.get("GEMINI_API_KEY", ""))
    secret_bin_key = force_clean(st.secrets.get("JSONBIN_API_KEY", ""))
    secret_bin_id = force_clean(st.secrets.get("JSONBIN_BIN_ID", ""))
    
    # Manual Override (ถ้าใส่ในช่อง Text จะใช้ค่านี้ก่อน)
    user_gemini = st.text_input("Gemini API Key (Manual Override)", type="password")
    
    # Logic เลือก Key: ถ้ามี Manual ให้ใช้ Manual, ถ้าไม่มีให้ใช้ Secret
    final_gemini_key = force_clean(user_gemini) if user_gemini else secret_gemini
    
    st.divider()
    
    # --- DEBUG SECTION (สำคัญมาก: ดูค่าที่นี่) ---
    with st.expander("🔴 DEBUG DATA (Check Here)", expanded=True):
        st.write(f"**Gemini Key Loaded:** {'YES' if final_gemini_key else 'NO'}")
        if final_gemini_key:
            st.code(f"Key Prefix: {final_gemini_key[:5]}...")
            st.code(f"Key Length: {len(final_gemini_key)}")
            
        st.write(f"**JSONBin Key:** {'YES' if secret_bin_key else 'NO'}")
        st.write(f"**JSONBin ID:** {'YES' if secret_bin_id else 'NO'}")
        if secret_bin_id:
            st.code(f"Bin ID Raw: >{secret_bin_id}<") # เช็คว่ามีช่องว่างไหม
            
        # สร้าง URL หลอกๆ เพื่อเช็ค Error
        test_url = f"[https://api.jsonbin.io/v3/b/](https://api.jsonbin.io/v3/b/){secret_bin_id}/latest"
        st.caption(f"Test URL: {test_url}")
    # -------------------------------------------

    # โหลด Database
    if "library" not in st.session_state:
        if secret_bin_key and secret_bin_id:
            data, error = get_prompts_safe(secret_bin_key, secret_bin_id)
            st.session_state.library = data
            st.session_state.db_error = error
        else:
            st.session_state.library = DEFAULT_PROMPTS
            st.session_state.db_error = "No Keys Provided"

    if st.session_state.db_error:
        st.error("DB Status: Error")
        st.caption(st.session_state.db_error)
        if st.button("Reload DB"):
            del st.session_state.library
            st.rerun()
    else:
        st.success(f"DB Status: OK ({len(st.session_state.library)} items)")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["✨ Gen Image", "🏷️ Bulk SEO", "📝 Writer", "📚 Library", "ℹ️ Models"])

# [Tab 1, 2, 3 ตัดออกเพื่อประหยัดพื้นที่ แต่ Logic เดิม]
# Copy Code ส่วน Tab 1-4 จากด้านบนมาใส่ได้เลยครับ หรือใช้แบบย่อเพื่อทดสอบก่อน
# ... (ใส่ Logic Gen Image เดิมตรงนี้ โดยใช้ final_gemini_key) ...

# === TAB 1: GENERATE IMAGE ===
with tab1:
    st.subheader("Generate Image")
    files = st.file_uploader("Upload", accept_multiple_files=True, key="t1_up")
    prompt = st.text_area("Prompt", "A gold ring...")
    if st.button("Generate"):
        if not final_gemini_key: st.error("No API Key!")
        else:
            imgs = [Image.open(f) for f in files] if files else []
            if not imgs: st.error("No Images")
            else:
                d, e = generate_image(final_gemini_key, imgs, prompt)
                if d: st.image(d)
                else: st.error(e)

# === TAB 5: MODELS (แก้ปัญหา Scan Failed) ===
with tab5:
    st.header("Check Models")
    if st.button("Scan Models"):
        if not final_gemini_key:
            st.error("❌ No API Key found. Please enter in Sidebar.")
        else:
            with st.spinner("Scanning..."):
                m, err = list_available_models(final_gemini_key)
                if m:
                    st.success(f"Found {len(m)} models")
                    st.dataframe([x['name'] for x in m if 'gemini' in x['name']])
                else:
                    st.error(f"Scan Failed: {err}")
                    # Debug URL
                    st.code(f"Failed URL was: [https://generativelanguage.googleapis.com/v1beta/models?key=](https://generativelanguage.googleapis.com/v1beta/models?key=){final_gemini_key}")

# ... (ส่วน Tab 2,3,4 ใช้ Logic เดียวกันคือเปลี่ยนไปใช้ final_gemini_key แทน api_key เดิม)
