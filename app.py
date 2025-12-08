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

# --- HELPER: SUPER CLEANER ---
def force_clean(value):
    if value is None: return ""
    s = str(value)
    s = s.replace("\n", "").replace("\r", "").replace("\t", "")
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

# --- 2. DATABASE FUNCTIONS ---
def get_prompts_safe(api_key, bin_id):
    try:
        if not api_key or not bin_id:
            return DEFAULT_PROMPTS, "Missing Keys"
        url = f"[https://api.jsonbin.io/v3/b/](https://api.jsonbin.io/v3/b/){bin_id}/latest"
        headers = {"X-Master-Key": api_key, "X-Bin-Meta": "false"}
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
        if res.status_code != 200: st.error(f"Save Failed: {res.text}")
    except Exception as e: st.error(f"Save Error: {e}")

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
st.title("Jewelry AI Studio (v2.1 Fix State)")

# *** SIDEBAR & KEY MANAGEMENT ***
with st.sidebar:
    st.title("💎 Config")
    
    secret_gemini = force_clean(st.secrets.get("GEMINI_API_KEY", ""))
    secret_bin_key = force_clean(st.secrets.get("JSONBIN_API_KEY", ""))
    secret_bin_id = force_clean(st.secrets.get("JSONBIN_BIN_ID", ""))
    
    user_gemini = st.text_input("Gemini API Key (Manual Override)", type="password")
    final_gemini_key = force_clean(user_gemini) if user_gemini else secret_gemini
    
    st.divider()
    
    with st.expander("🔴 DEBUG DATA"):
        st.write(f"Gemini Key: {'OK' if final_gemini_key else 'Missing'}")
        if secret_bin_id: st.code(f"Bin ID: >{secret_bin_id}<")
    
    # --- FIX 2.1: INITIALIZE DB STATE FIRST ---
    if "db_error" not in st.session_state: st.session_state.db_error = None
    if "library" not in st.session_state: st.session_state.library = None
    
    # Load Database if empty
    if st.session_state.library is None:
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
            st.session_state.library = None
            st.rerun()
    else:
        st.success(f"DB Status: OK ({len(st.session_state.library)} items)")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["✨ Gen Image", "🏷️ Bulk SEO", "📝 Writer", "📚 Library", "ℹ️ Models"])

# === TAB 1: GENERATE IMAGE ===
with tab1:
    st.subheader("Generate Image")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        files = st.file_uploader("Upload", accept_multiple_files=True, key="t1_up")
        imgs = [Image.open(f) for f in files] if files else []
        if imgs: 
            st.caption(f"{len(imgs)} selected")
            cols = st.columns(3)
            for i, im in enumerate(imgs): cols[i%3].image(im, use_column_width=True)
            
    with c2:
        lib = st.session_state.library
        cats = list(set(p.get('category', 'Other') for p in lib)) if lib else []
        sel_cat = st.selectbox("Category", cats) if cats else None
        filtered = [p for p in lib if p.get('category') == sel_cat]
        
        prompt_val = "A gold ring..."
        if filtered:
            sel_style = st.selectbox("Style", filtered, format_func=lambda x: x.get('name', 'Unknown'))
            if sel_style:
                if sel_style.get("sample_url"): safe_st_image(sel_style["sample_url"], width=80)
                vars_list = [v.strip() for v in sel_style.get('variables', '').split(",") if v.strip()]
                user_vals = {v: st.text_input(v) for v in vars_list}
                prompt_val = sel_style.get('template', '')
                for k, v in user_vals.items(): prompt_val = prompt_val.replace(f"{{{k}}}", v)
        
        prompt = st.text_area("Prompt", prompt_val, height=100)
        
        if st.button("Generate", type="primary"):
            if not final_gemini_key: st.error("No API Key!")
            elif not imgs: st.error("No Images")
            else:
                with st.spinner("Generating..."):
                    d, e = generate_image(final_gemini_key, imgs, prompt)
                    if d: 
                        st.session_state.last_gen = d
                        st.rerun()
                    else: st.error(e)
                    
        if "last_gen" in st.session_state:
            st.image(st.session_state.last_gen)

# === TAB 2: BULK SEO ===
with tab2:
    st.header("Bulk SEO Tags")
    bc1, bc2 = st.columns([1, 1.5])
    with bc1:
        files = st.file_uploader("Upload Images", accept_multiple_files=True, key="bulk_up")
        imgs = [Image.open(f) for f in files] if files else []
    with bc2:
        url = st.text_input("Product URL:", key="bulk_url")
        if st.button("Run Batch", type="primary"):
            if not final_gemini_key or not url: st.error("Check Key & URL")
            else:
                pbar = st.progress(0); res_area = st.container()
                for i, img in enumerate(imgs):
                    with st.spinner(f"#{i+1}..."):
                        json_txt, err = generate_seo_for_existing_image(final_gemini_key, img, url)
                        pbar.progress((i+1)/len(imgs))
                        with res_area:
                            c1, c2 = st.columns([1, 4])
                            c1.image(img, width=80)
                            if json_txt:
                                d = parse_json_response(json_txt)
                                if d:
                                    with c2.expander(f"✅ #{i+1}", expanded=True):
                                        st.code(clean_filename(d.get('file_name')), language="text")
                                        st.code(d.get('alt_tag'), language="text")
                                else: c2.code(json_txt)
                            else: c2.error(err)

# === TAB 3: WRITER ===
with tab3:
    st.header("Product Writer")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        files = st.file_uploader("Images", accept_multiple_files=True, key="w_img")
        writer_imgs = [Image.open(f) for f in files] if files else []
        raw = st.text_area("Paste Details:", height=300)
        if st.button("Generate Content", type="primary"):
            if not final_gemini_key or not raw: st.error("Missing Info")
            else:
                with st.spinner("Writing..."):
                    json_txt, err = generate_full_product_content(final_gemini_key, writer_imgs, raw)
                    if json_txt:
                        st.session_state.last_write = json_txt
                        st.rerun()
                    else: st.error(err)
    with c2:
        if "last_write" in st.session_state:
            d = parse_json_response(st.session_state.last_write)
            if d:
                st.code(d.get('meta_title'), language="text")
                st.code(d.get('meta_description'), language="text")
                with st.expander("HTML"): st.code(d.get('html_content'), language="html")
                st.markdown(d.get('html_content', ''), unsafe_allow_html=True)

# === TAB 4: LIBRARY ===
with tab4:
    st.subheader("Library Manager")
    target = st.session_state.get('edit_target')
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
            save_prompts_safe(st.session_state.library, secret_bin_key, secret_bin_id)
            st.session_state.edit_target = None
            st.rerun()
            
    st.divider()
    for i, p in enumerate(st.session_state.library):
        c1, c2, c3, c4 = st.columns([1, 4, 1, 1])
        if p.get("sample_url"): safe_st_image(p["sample_url"], width=50)
        c2.write(f"**{p.get('name')}**")
        if c3.button("✏️", key=f"e{i}"): st.session_state.edit_target = p; st.rerun()
        if c4.button("🗑️", key=f"d{i}"): st.session_state.library.pop(i); save_prompts_safe(st.session_state.library, secret_bin_key, secret_bin_id); st.rerun()

# === TAB 5: MODELS ===
with tab5:
    st.header("Check Models")
    if st.button("Scan Models"):
        if not final_gemini_key:
            st.error("No API Key")
        else:
            with st.spinner("Scanning..."):
                m, err = list_available_models(final_gemini_key)
                if m:
                    st.success(f"Found {len(m)} models")
                    st.dataframe([x['name'] for x in m if 'gemini' in x['name']])
                else:
                    st.error(f"Scan Failed: {err}")
                    st.code(f"Key used: {final_gemini_key}")
