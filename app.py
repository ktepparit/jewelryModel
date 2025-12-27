import streamlit as st
import json
import requests
import base64
from io import BytesIO
from PIL import Image
import time
import pandas as pd
import re
import zipfile

# --- 1. CONFIGURATION & CONSTANTS ---
st.set_page_config(layout="wide", page_title="Jewelry AI Studio 12/9")

# Model IDs
MODEL_IMAGE_GEN = "models/gemini-3-pro-image-preview"
MODEL_TEXT_GEMINI = "models/gemini-3-pro-preview"
MODEL_TEXT_CLAUDE = "claude-sonnet-4-20250514"

# --- HELPER: CLEANER ---
def clean_key(value):
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

SEO_PROMPT_SMART_GEN = """
You are an SEO & Visual Content Specialist for Jewelry e-commerce.
Your task is to generate an SEO-optimized **Image File Name** and **Alt Tag** based on the visual description and product context provided.

**Inputs:**
1. **Visual Instruction (The image is generated from this):** "{context}"
2. **Product Reference URL (Context):** "{product_url}"

**Instructions:**
- **File Name:** Create a lowercase, hyphenated file name ending in .jpg (e.g., `silver-ring-blue-gemstone-side-view.jpg`).
    - COMBINE keywords from the URL (if valid) with the VISUAL details from the instruction.
    - Do NOT simply copy the URL slug. The filename MUST describe the visual look of the image (e.g., pose, angle, lighting, material).
- **Alt Tag:** Write a natural English sentence describing the image for accessibility and SEO. Mention the material, stone, and style visible in the instruction.

IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure: {"file_name": "...", "alt_tag": "..."}
"""

SEO_PROMPT_BULK_EXISTING = """
คุณคือ SEO specialist ที่มีประสบการณ์ 15-20 ปี ช่วยเขียน SEO-optimized image file name with image alt tags เป็นภาษาอังกฤษ สำหรับสินค้าของฉันตามแต่ละรูปที่แนบมาให้ {product_url} เพื่อให้ได้ติดอันดับที่ดีบน organic search engine โดยกลุ่มลูกค้าเป็นผู้สนใจสินค้าชนิดนี้
IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure: {"file_name": "...", "alt_tag": "..."}
"""

SEO_PRODUCT_WRITER_PROMPT = """
คุณมีหน้าที่เป็นผู้เชี่ยวชาญ SEO specialist product content writer ผู้มีประสบการ์ 15-20 ปี ช่วยเขียน SEO-Optimized product description เป็นภาษาอังกฤษสำหรับร้าน e-commerce ของฉันที่สร้างโดยShopify ตามโครงสร้าง <h1><h2> with human-written style that pass AI Content detector app https://undetectable.ai 

เป้าหมายเพื่อเพิ่มอันดับบน organic search engine และ AI search แนะนำ product ของฉันให้กับลูกค้า

ให้คุณแบ่งการวางคีย์เวิร์ดตามโครงสร้างของ Product Description ดังนี้:

1. ย่อหน้าแรก (Opening Paragraph) - คีย์เวิร์ดหลัก + Semantic Keyword 1-2 คำ
2. ส่วนกลางของเนื้อหา (Body) - กระจาย Semantic Keywords
3. ส่วนคุณสมบัติ (Specifications) - วัสดุ เช่น 925 sterling silver, handcrafted
4. ส่วน FAQ - Long-tail keywords

โครงสร้าง: Product Overview, Key Features, FAQ

รวมทั้งช่วยเขียน:
- Google SEO-optimized meta title (approximately 60 characters)
- Google SEO-optimized meta description
- SEO-optimized image file name + alt tag สำหรับทุก images
- URL slug

Input Data: {raw_input}

IMPORTANT OUTPUT FORMAT:
You MUST return the result in RAW JSON format ONLY. Do not include markdown backticks.
{
  "url_slug": "url-slug-example",
  "meta_title": "Meta Title Example (Max 60 chars)",
  "meta_description": "Meta Description Example (Max 160 chars)",
  "product_title_h1": "Product Title Example",
  "html_content": "<p>Your full HTML product description here...</p>",
  "image_seo": [
    { "file_name": "image-name.jpg", "alt_tag": "Image description" }
  ]
}
"""

SEO_PROMPT_NAME_SLUG = """
You are an SEO expert with 10-15 years of experience. 
Analyze the provided product images and description. Generate:
1. An attractive, SEO-optimized Product Name.
2. A suitable, clean URL Slug (using hyphens).

User Input Description: "{user_desc}"

IMPORTANT: Return RAW JSON format ONLY (no markdown backticks).
Structure: {"product_name": "...", "url_slug": "..."}
"""

# Default Data
DEFAULT_PROMPTS = [
    {"id": "p1", "name": "Luxury Hand (Ring)", "category": "Ring",
     "template": "A realistic close-up of a female hand model wearing a ring with {face_size} face size, soft studio lighting, elegant jewelry photography.",
     "variables": "face_size", "sample_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Ring_render.jpg/320px-Ring_render.jpg"},
    {"id": "rt1", "name": "Clean Studio Look", "category": "Retouch",
     "template": "Retouch this jewelry product to have a clean white studio background. Enhance the metal shine of {metal_type} and gemstone clarity. Professional product photography.",
     "variables": "metal_type", "sample_url": ""}
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
    except: return DEFAULT_PROMPTS

def save_prompts(data):
    try:
        raw_key = st.secrets.get("JSONBIN_API_KEY", "")
        raw_bin = st.secrets.get("JSONBIN_BIN_ID", "")
        API_KEY = clean_key(raw_key)
        BIN_ID = clean_key(raw_bin)
        url = f"https://api.jsonbin.io/v3/b/{BIN_ID}"
        headers = {"Content-Type": "application/json", "X-Master-Key": API_KEY}
        requests.put(url, json=data, headers=headers, timeout=10)
    except Exception as e: st.error(f"Save failed: {e}")

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

def remove_html_tags(text):
    if not text: return ""
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<')
    return "\n".join([line.strip() for line in text.split('\n') if line.strip()])

# --- SHOPIFY HELPER FUNCTIONS ---
def update_shopify_product_v2(shop_url, access_token, product_id, data, images_pil=None, upload_images=False):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    product_payload = {
        "id": product_id,
        "title": data.get('product_title_h1'),
        "body_html": data.get('html_content'),
        "metafields": [
            {"namespace": "global", "key": "title_tag", "value": data.get('meta_title', ''), "type": "single_line_text_field"},
            {"namespace": "global", "key": "description_tag", "value": data.get('meta_description', ''), "type": "multi_line_text_field"}
        ]
    }
    
    if upload_images and images_pil and "image_seo" in data:
        img_payloads = []
        image_seo_list = data.get("image_seo", [])
        for i, img in enumerate(images_pil):
            seo_info = image_seo_list[i] if i < len(image_seo_list) else {}
            img_payloads.append({
                "attachment": img_to_base64(img),
                "filename": seo_info.get("file_name", f"image_{i+1}.jpg"),
                "alt": seo_info.get("alt_tag", "")
            })
        if img_payloads: product_payload["images"] = img_payloads

    try:
        response = requests.put(url, json={"product": product_payload}, headers=headers)
        if response.status_code in [200, 201]: return True, "✅ Update Successful!"
        return False, f"Shopify API Error {response.status_code}: {response.text}"
    except Exception as e: return False, f"Connection Error: {str(e)}"

def add_single_image_to_shopify(shop_url, access_token, product_id, image_bytes, file_name=None, alt_tag=None):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}/images.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    if not image_bytes: return False, "No valid image data."
    b64_str = base64.b64encode(image_bytes).decode('utf-8')
    payload = {"image": {"attachment": b64_str, "filename": file_name or f"gen_ai_image_{int(time.time())}.jpg", "alt": alt_tag or "AI Generated Product Image"}}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code in [200, 201]: return True, "✅ Added Successful!"
        return False, f"Shopify Error {response.status_code}: {response.text}"
    except Exception as e: return False, f"Connection Error: {str(e)}"

def upload_only_images_to_shopify(shop_url, access_token, product_id, image_bytes_list):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    img_payloads = []
    for i, img_bytes in enumerate(image_bytes_list):
        if img_bytes:
            img_payloads.append({"attachment": base64.b64encode(img_bytes).decode('utf-8'), "filename": f"retouched_image_{i+1}.jpg", "alt": f"Retouched Product Image {i+1}"})
    if not img_payloads: return False, "No valid images to upload."
    
    try:
        response = requests.put(url, json={"product": {"id": product_id, "images": img_payloads}}, headers=headers)
        if response.status_code in [200, 201]: return True, "✅ Upload Successful!"
        return False, f"Shopify Error {response.status_code}: {response.text}"
    except Exception as e: return False, f"Connection Error: {str(e)}"

def get_shopify_product_images(shop_url, access_token, product_id):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}/images.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            pil_images = []
            for img_info in response.json().get("images", []):
                src = img_info.get("src")
                if src:
                    img_resp = requests.get(src, stream=True)
                    if img_resp.status_code == 200:
                        img_pil = Image.open(BytesIO(img_resp.content))
                        if img_pil.mode in ('RGBA', 'P'): img_pil = img_pil.convert('RGB')
                        pil_images.append(img_pil)
            return pil_images, None
        return None, f"Shopify API Error {response.status_code}: {response.text}"
    except Exception as e: return None, f"Connection Error: {str(e)}"

def get_shopify_product_details(shop_url, access_token, product_id):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"): shop_url += ".myshopify.com"
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            prod = response.json().get("product", {})
            return prod.get("body_html", ""), prod.get("title", ""), prod.get("handle", ""), None
        return None, None, None, f"Error {response.status_code}: {response.text}"
    except Exception as e: return None, None, None, str(e)

# ============================================================
# --- CLAUDE API FUNCTION (NEW) ---
# ============================================================
def call_claude_api(claude_key, prompt, img_pil_list=None):
    """Call Claude API for Text/SEO tasks with optional image support"""
    url = "https://api.anthropic.com/v1/messages"
    headers = {"Content-Type": "application/json", "x-api-key": claude_key, "anthropic-version": "2023-06-01"}
    
    content = []
    if img_pil_list:
        for img in img_pil_list:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_to_base64(img)}})
    content.append({"type": "text", "text": prompt})
    
    payload = {"model": MODEL_TEXT_CLAUDE, "max_tokens": 4096, "messages": [{"role": "user", "content": content}]}
    
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=120)
            if res.status_code == 200:
                text_content = ""
                for block in res.json().get("content", []):
                    if block.get("type") == "text": text_content += block.get("text", "")
                return text_content, None
            elif res.status_code == 529: time.sleep(3); continue
            else: return None, f"Claude API Error {res.status_code}: {res.text}"
        except: time.sleep(2)
    return None, "Claude API failed after retries"

# ============================================================
# --- AI FUNCTIONS (GEMINI & CLAUDE) ---
# ============================================================
def generate_image(api_key, image_list, prompt):
    """Image Generation - Gemini Only"""
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_IMAGE_GEN}:generateContent?key={key}"
    full_prompt = f"Instruction: {prompt} \nImportant Constraint: Keep the main jewelry product in the input image EXACTLY as it looks. Only improve lighting, background, and photography quality."
    
    parts = [{"text": full_prompt}]
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

def generate_seo_tags_smart(gemini_key, claude_key, selected_model, context, product_url=""):
    prompt = SEO_PROMPT_SMART_GEN.replace("{context}", context).replace("{product_url}", product_url)
    if selected_model == "Claude" and claude_key: return call_claude_api(claude_key, prompt)
    key = clean_key(gemini_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_TEXT_GEMINI}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.5, "responseMimeType": "application/json"}}
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.status_code == 200: return res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text"), None
            elif res.status_code == 503: time.sleep(2); continue
            else: return None, f"Error {res.status_code}"
        except: time.sleep(1)
    return None, "Failed"

def generate_seo_for_existing_image(gemini_key, claude_key, selected_model, img_pil, product_url):
    prompt = SEO_PROMPT_BULK_EXISTING.replace("{product_url}", product_url)
    if selected_model == "Claude" and claude_key: return call_claude_api(claude_key, prompt, [img_pil])
    key = clean_key(gemini_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_TEXT_GEMINI}:generateContent?key={key}"
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}}]}], "generationConfig": {"temperature": 0.5, "responseMimeType": "application/json"}}
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.status_code == 200: return res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text"), None
            elif res.status_code == 503: time.sleep(2); continue
            else: return None, f"Error {res.status_code}"
        except: time.sleep(1)
    return None, "Failed"

def generate_full_product_content(gemini_key, claude_key, selected_model, img_pil_list, raw_input):
    prompt = SEO_PRODUCT_WRITER_PROMPT.replace("{raw_input}", raw_input)
    num_images = len(img_pil_list) if img_pil_list else 0
    if num_images > 0: prompt += f"\n\nCRITICAL: You received {num_images} images. Return exactly {num_images} objects in 'image_seo' array."
    if selected_model == "Claude" and claude_key: return call_claude_api(claude_key, prompt, img_pil_list)
    key = clean_key(gemini_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_TEXT_GEMINI}:generateContent?key={key}"
    parts = [{"text": prompt}]
    if img_pil_list:
        for img in img_pil_list: parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}}
    for attempt in range(3):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
            if res.status_code == 200: return res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text"), None
            elif res.status_code == 503: time.sleep(3); continue
            else: return None, f"Error {res.status_code}: {res.text}"
        except: time.sleep(1)
    return None, "Failed"

def generate_seo_name_slug(gemini_key, claude_key, selected_model, img_list, user_desc):
    prompt = SEO_PROMPT_NAME_SLUG.replace("{user_desc}", user_desc)
    pil_images = []
    if img_list:
        for item in img_list:
            if isinstance(item, bytes):
                try: pil_images.append(Image.open(BytesIO(item)))
                except: pass
            elif isinstance(item, Image.Image): pil_images.append(item)
    if selected_model == "Claude" and claude_key: return call_claude_api(claude_key, prompt, pil_images if pil_images else None)
    key = clean_key(gemini_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_TEXT_GEMINI}:generateContent?key={key}"
    parts = [{"text": prompt}]
    for img in pil_images: parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}}
    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if res.status_code == 200: return res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text"), None
        return None, f"Error {res.status_code}"
    except Exception as e: return None, str(e)

def list_available_models(api_key):
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200: return response.json().get("models", [])
        return None
    except: return None

# ============================================================
# --- UI LOGIC ---
# ============================================================
if "library" not in st.session_state: st.session_state.library = get_prompts()
if "edit_target" not in st.session_state: st.session_state.edit_target = None
if "image_generated_success" not in st.session_state: st.session_state.image_generated_success = False
if "current_generated_image" not in st.session_state: st.session_state.current_generated_image = None
if "gen_tags_result" not in st.session_state: st.session_state.gen_tags_result = {}
if "bulk_results" not in st.session_state: st.session_state.bulk_results = None
if "writer_result" not in st.session_state: st.session_state.writer_result = None
if "retouch_results" not in st.session_state: st.session_state.retouch_results = None
if "seo_name_result" not in st.session_state: st.session_state.seo_name_result = None
if "bulk_key_counter" not in st.session_state: st.session_state.bulk_key_counter = 0
if "writer_key_counter" not in st.session_state: st.session_state.writer_key_counter = 0
if "retouch_key_counter" not in st.session_state: st.session_state.retouch_key_counter = 0

# ============================================================
# --- SIDEBAR (WITH MODEL SELECTOR) ---
# ============================================================
with st.sidebar:
    st.title("⚙️ Config")
    
    # MODEL SELECTOR
    st.subheader("🤖 AI Model Selection")
    selected_text_model = st.selectbox("Text/SEO Model:", ["Gemini", "Claude"], index=0, help="เลือก Model สำหรับงาน SEO Writing", key="sidebar_model_select")
    st.session_state['selected_text_model'] = selected_text_model
    st.caption("📸 Image Gen: Gemini (Fixed)")
    st.divider()
    
    # API KEYS
    st.subheader("🔑 API Keys")
    
    # Gemini Key
    if "GEMINI_API_KEY" in st.secrets:
        gemini_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini Key Loaded")
    elif "GOOGLE_API_KEY" in st.secrets:
        gemini_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ Google Key Loaded")
    else:
        gemini_key = st.text_input("Gemini API Key", type="password", key="sidebar_gemini_key")
    gemini_key = clean_key(gemini_key)
    
    # Claude Key
    if "CLAUDE_API_KEY" in st.secrets:
        claude_key = clean_key(st.secrets["CLAUDE_API_KEY"])
        st.success("✅ Claude Key Loaded")
    else:
        claude_key = st.text_input("Claude API Key (Optional)", type="password", key="sidebar_claude_key")
        claude_key = clean_key(claude_key)
    
    if selected_text_model == "Claude" and not claude_key:
        st.warning("⚠️ Claude selected but no API Key!")
    
    st.divider()
    if "JSONBIN_API_KEY" in st.secrets: st.caption("✅ Database Connected")
    else: st.warning("⚠️ Local Mode")
    st.divider()
    st.caption(f"**Active Text Model:** {selected_text_model}")
    st.caption(f"**Active Image Model:** Gemini")

st.title("💎 Jewelry AI Studio")
tab1, tab_retouch, tab2, tab3, tab4, tab5 = st.tabs(["✨ Gen Image", "🎨 Retouch", "🏷️ Bulk SEO", "📝 Writer", "📚 Library", "ℹ️ Models"])

# === TAB 1: GEN IMAGE ===
with tab1:
    if "gen_shopify_imgs" not in st.session_state: st.session_state.gen_shopify_imgs = []
    if "gen_key_counter" not in st.session_state: st.session_state.gen_key_counter = 0
    
    gen_key_id = st.session_state.gen_key_counter
    
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        st.subheader("1. Source Images")
        with st.expander("🛍️ Import from Shopify", expanded=True):
            sh_secret_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            sh_secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            if sh_secret_shop and sh_secret_token:
                sh_gen_id = st.text_input("Product ID", key=f"gen_shopify_id_{gen_key_id}")
                col_fetch, col_clear = st.columns([2, 1])
                if col_fetch.button("⬇️ Fetch Images", key=f"gen_fetch_btn_{gen_key_id}"):
                    if not sh_gen_id: st.warning("Enter ID")
                    else:
                        with st.spinner("Downloading..."):
                            imgs, err = get_shopify_product_images(sh_secret_shop, sh_secret_token, sh_gen_id)
                            if imgs:
                                _, _, handle, _ = get_shopify_product_details(sh_secret_shop, sh_secret_token, sh_gen_id)
                                if handle:
                                    clean_shop = sh_secret_shop.replace("https://", "").replace("http://", "").strip()
                                    if not clean_shop.endswith(".myshopify.com"): clean_shop += ".myshopify.com"
                                    # Save URL directly to the widget key
                                    product_url = f"https://{clean_shop.replace('.myshopify.com', '.com')}/products/{handle}"
                                    st.session_state[f"gen_post_url_{gen_key_id}"] = product_url
                                st.session_state.gen_shopify_imgs = imgs
                                st.session_state['gen_upload_id'] = sh_gen_id
                                st.success(f"Loaded {len(imgs)} images"); st.rerun()
                            else: st.error(err)
                if col_clear.button("❌ Clear", key=f"gen_clear_btn_{gen_key_id}"):
                    st.session_state.gen_shopify_imgs = []
                    st.session_state.image_generated_success = False
                    st.session_state.current_generated_image = None
                    st.session_state.gen_tags_result = {}
                    # Clear URL by incrementing key counter (creates new widget)
                    st.session_state.gen_key_counter += 1
                    st.rerun()
            else: st.info("Set Secrets to use Import")

        images_to_send = []
        if st.session_state.gen_shopify_imgs:
            images_to_send = st.session_state.gen_shopify_imgs
            st.info(f"Using {len(images_to_send)} images from Shopify")
            try:
                zip_gen = BytesIO()
                with zipfile.ZipFile(zip_gen, "w") as zf:
                    for i, img in enumerate(images_to_send):
                        buf = BytesIO(); img.save(buf, format="JPEG", quality=95)
                        zf.writestr(f"shopify_orig_{i+1}.jpg", buf.getvalue())
                st.download_button("💾 Download All Originals (.zip)", data=zip_gen.getvalue(), file_name="shopify_original_images.zip", mime="application/zip", key=f"gen_download_zip_{gen_key_id}")
            except: pass
        else:
            files = st.file_uploader("Upload Manual", accept_multiple_files=True, type=["jpg","png"], key=f"gen_up_{gen_key_id}")
            images_to_send = [Image.open(f) for f in files] if files else []
        if images_to_send:
            cols = st.columns(4)
            for i, img in enumerate(images_to_send): cols[i%4].image(img, use_column_width=True)

    with c2:
        st.subheader("2. Settings")
        current_text_model = st.session_state.get('selected_text_model', 'Gemini')
        st.caption(f"🤖 SEO Tags Model: **{current_text_model}**")
        lib = st.session_state.library
        cats = list(set(p.get('category','Other') for p in lib)) if lib else []
        sel_cat = st.selectbox("Category", cats, key=f"gen_cat_{gen_key_id}") if cats else None
        filtered = [p for p in lib if p.get('category') == sel_cat]
        if filtered:
            sel_style = st.selectbox("Style", filtered, format_func=lambda x: x.get('name','Unknown'), key=f"gen_style_{gen_key_id}")
            if sel_style.get("sample_url"): safe_st_image(sel_style["sample_url"], width=100)
            
            # Get current style id
            current_style_id = sel_style.get('id', 'default')
            
            # Widget key for the text_area
            prompt_widget_key = f"gen_prompt_display_{gen_key_id}_{current_style_id}"
            
            # Key to track last variable values
            vars_tracker_key = f"gen_vars_tracker_{gen_key_id}_{current_style_id}"
            
            # Initialize with template if widget key not exists
            if prompt_widget_key not in st.session_state:
                st.session_state[prompt_widget_key] = sel_style.get('template','')
            
            # Variables input
            vars_list = [v.strip() for v in sel_style.get('variables','').split(",") if v.strip()]
            
            if vars_list:
                st.write("**Variables:**")
                cols_vars = st.columns(len(vars_list)) if len(vars_list) <= 3 else st.columns(3)
                user_vals = {}
                for idx, v in enumerate(vars_list):
                    with cols_vars[idx % len(cols_vars)]:
                        user_vals[v] = st.text_input(v, key=f"gen_var_{v}_{gen_key_id}_{current_style_id}")
                
                # Check if variables changed from last time
                last_vars = st.session_state.get(vars_tracker_key, {})
                vars_changed = (user_vals != last_vars)
                
                # Only update prompt if variables changed
                if vars_changed:
                    current_prompt = sel_style.get('template','')
                    for k, val in user_vals.items(): 
                        current_prompt = current_prompt.replace(f"{{{k}}}", val)
                    st.session_state[prompt_widget_key] = current_prompt
                    st.session_state[vars_tracker_key] = user_vals.copy()
            
            # Editable text area - value comes from session state key automatically
            prompt_edit = st.text_area("✏️ Custom Instruction (edit if needed)", height=100, key=prompt_widget_key)
            
            # Product URL - auto-filled when Fetch from Shopify (value set directly to session state key)
            url_input = st.text_input("Product URL (Optional):", key=f"gen_post_url_{gen_key_id}", help="Auto-filled from Shopify. AI will use URL context for tags")

            if st.button("🚀 GENERATE", type="primary", use_container_width=True, key=f"gen_run_btn_{gen_key_id}"):
                if not gemini_key or not images_to_send: st.error("Check Key & Images")
                else:
                    with st.spinner("Generating Image & Smart Tags..."):
                        d, e = generate_image(gemini_key, images_to_send, prompt_edit)
                        if d:
                            st.session_state.current_generated_image = d
                            st.session_state.image_generated_success = True
                            current_url = url_input
                            tags_json, _ = generate_seo_tags_smart(gemini_key, claude_key, current_text_model, prompt_edit, current_url)
                            if tags_json:
                                parsed_tags = parse_json_response(tags_json)
                                st.session_state.gen_tags_result = parsed_tags if parsed_tags else {}
                            else: st.session_state.gen_tags_result = {}
                            st.rerun()
                        else: st.error(e)

            if st.session_state.image_generated_success and st.session_state.current_generated_image:
                st.divider(); st.subheader("✨ Result")
                st.image(st.session_state.current_generated_image, use_column_width=True)
                st.download_button("💾 Download Image", st.session_state.current_generated_image, "gen.jpg", "image/jpeg", type="secondary", key="gen_dl_img_tab1")
                st.divider(); st.subheader("☁️ Upload to Shopify")
                with st.container(border=True):
                    tags_data = st.session_state.get("gen_tags_result", {})
                    col_tags1, col_tags2 = st.columns(2)
                    final_filename = col_tags1.text_input("File Name", value=tags_data.get("file_name", ""), key="gen_filename_tab1")
                    final_alt = col_tags2.text_input("Alt Tag", value=tags_data.get("alt_tag", ""), key="gen_alt_tab1")
                    s_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
                    s_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
                    default_id = st.session_state.get("gen_shopify_id", "")
                    col_u1, col_u2 = st.columns([3, 1])
                    u_prod_id = col_u1.text_input("Product ID", value=default_id, key="gen_upload_id", label_visibility="collapsed")
                    if col_u2.button("🚀 Upload", type="primary", use_container_width=True, key="gen_upload_btn_tab1"):
                        if not s_shop or not s_token: st.error("Missing Shopify Secrets")
                        elif not u_prod_id: st.warning("Enter Product ID")
                        else:
                            with st.spinner("Uploading..."):
                                success, msg = add_single_image_to_shopify(s_shop, s_token, u_prod_id, st.session_state.current_generated_image, file_name=final_filename, alt_tag=final_alt)
                                if success: st.success(msg)
                                else: st.error(msg)

# === TAB RETOUCH ===
with tab_retouch:
    st.header("🎨 Retouch (via Gemini)")
    if "shopify_fetched_imgs" not in st.session_state: st.session_state.shopify_fetched_imgs = []
    rt_key_id = st.session_state.retouch_key_counter
    rt_c1, rt_c2 = st.columns([1, 1.2])
    
    with rt_c1:
        st.subheader("1. Input Images")
        with st.expander("🛍️ Import from Shopify", expanded=True):
            sh_secret_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            sh_secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            if sh_secret_shop and sh_secret_token:
                st.success("✅ Shopify Connected")
                sh_imp_id = st.text_input("Product ID to Fetch", key=f"rt_imp_id_{rt_key_id}")
                c_fetch, c_clear = st.columns([2,1])
                if c_fetch.button("⬇️ Fetch Images", key=f"rt_fetch_btn_{rt_key_id}"):
                    if not sh_imp_id: st.warning("Enter ID")
                    else:
                        with st.spinner("Downloading..."):
                            imgs, err = get_shopify_product_images(sh_secret_shop, sh_secret_token, sh_imp_id)
                            if imgs:
                                st.session_state.shopify_fetched_imgs = imgs
                                st.session_state['rt_upload_id'] = sh_imp_id
                                st.success(f"Loaded {len(imgs)} images!"); st.rerun()
                            else: st.error(err)
                if c_clear.button("❌ Clear", key=f"rt_clear_btn_{rt_key_id}"):
                    st.session_state.shopify_fetched_imgs = []
                    if 'rt_upload_id' in st.session_state: del st.session_state['rt_upload_id']
                    st.rerun()
            else: st.info("Set Secrets to use.")
        
        rt_imgs, source_type = [], ""
        if st.session_state.shopify_fetched_imgs:
            rt_imgs = st.session_state.shopify_fetched_imgs
            source_type = "Shopify"
            st.info(f"Using {len(rt_imgs)} images from Shopify")
            try:
                zip_orig = BytesIO()
                with zipfile.ZipFile(zip_orig, "w") as zf:
                    for i, img in enumerate(rt_imgs):
                        buf = BytesIO(); img.save(buf, format="JPEG", quality=95)
                        zf.writestr(f"original_{i+1}.jpg", buf.getvalue())
                st.download_button("💾 Download Originals (.zip)", data=zip_orig.getvalue(), file_name="originals.zip", mime="application/zip", key=f"rt_dl_orig_{rt_key_id}")
            except: pass
        else:
            rt_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png"], key=f"rt_up_{rt_key_id}")
            if rt_files: rt_imgs = [Image.open(f) for f in rt_files]; source_type = "Upload"
        if rt_imgs:
            with st.expander(f"📸 View Input ({len(rt_imgs)} images)", expanded=False):
                cols = st.columns(4)
                for i, img in enumerate(rt_imgs): cols[i%4].image(img, use_column_width=True)
        else: st.warning("Waiting for images...")

    with rt_c2:
        st.subheader("2. Prompt Settings")
        lib = st.session_state.library
        rt_cats = list(set(p.get('category','Other') for p in lib)) if lib else []
        default_cat_index = rt_cats.index("Retouch") if "Retouch" in rt_cats else 0
        rt_sel_cat = st.selectbox("Category", rt_cats, index=default_cat_index, key=f"rt_cat_{rt_key_id}") if rt_cats else None
        rt_filtered = [p for p in lib if p.get('category') == rt_sel_cat]
        if rt_filtered:
            rt_style = st.selectbox("Style", rt_filtered, format_func=lambda x: x.get('name','Unknown'), key=f"rt_style_{rt_key_id}")
            
            # Get current style id for dynamic key
            current_rt_style_id = rt_style.get('id', 'default')
            
            # Widget key for the text_area
            rt_prompt_widget_key = f"rt_prompt_display_{rt_key_id}_{current_rt_style_id}"
            
            # Key to track last variable values
            rt_vars_tracker_key = f"rt_vars_tracker_{rt_key_id}_{current_rt_style_id}"
            
            # Initialize with template if widget key not exists
            if rt_prompt_widget_key not in st.session_state:
                st.session_state[rt_prompt_widget_key] = rt_style.get('template','')
            
            # Variables input
            rt_vars = [v.strip() for v in rt_style.get('variables','').split(",") if v.strip()]
            
            if rt_vars:
                st.write("**Variables:**")
                cols_vars = st.columns(len(rt_vars)) if len(rt_vars) <= 3 else st.columns(3)
                rt_user_vals = {}
                for idx, v in enumerate(rt_vars):
                    with cols_vars[idx % len(cols_vars)]:
                        rt_user_vals[v] = st.text_input(v, key=f"rt_var_{v}_{rt_key_id}_{current_rt_style_id}")
                
                # Check if variables changed from last time
                last_rt_vars = st.session_state.get(rt_vars_tracker_key, {})
                rt_vars_changed = (rt_user_vals != last_rt_vars)
                
                # Only update prompt if variables changed
                if rt_vars_changed:
                    current_rt_prompt = rt_style.get('template','')
                    for k, val in rt_user_vals.items(): 
                        current_rt_prompt = current_rt_prompt.replace(f"{{{k}}}", val)
                    st.session_state[rt_prompt_widget_key] = current_rt_prompt
                    st.session_state[rt_vars_tracker_key] = rt_user_vals.copy()
            
            # Editable text area - value comes from session state key automatically
            rt_prompt_edit = st.text_area("✏️ Custom Instruction (edit if needed)", height=100, key=rt_prompt_widget_key)
            
            c_rt1, c_rt2 = st.columns([1, 1])
            run_retouch = c_rt1.button("🚀 Run Batch", type="primary", disabled=(not rt_imgs), key=f"rt_run_btn_{rt_key_id}")
            clear_retouch = c_rt2.button("🔄 Start Over", key=f"rt_startover_btn_{rt_key_id}")
            if clear_retouch:
                st.session_state.retouch_results = None; st.session_state.seo_name_result = None
                st.session_state.shopify_fetched_imgs = []; st.session_state.retouch_key_counter += 1
                if 'rt_upload_id' in st.session_state: del st.session_state['rt_upload_id']
                st.rerun()
            if run_retouch:
                if not gemini_key: st.error("Missing Gemini API Key!")
                else:
                    rt_temp_results = []
                    rt_pbar = st.progress(0)
                    for i, img in enumerate(rt_imgs):
                        with st.spinner(f"Processing #{i+1}..."):
                            gen_img_bytes, err = generate_image(gemini_key, [img], rt_prompt_edit)
                            rt_pbar.progress((i+1)/len(rt_imgs))
                            rt_temp_results.append(gen_img_bytes if gen_img_bytes else None)
                            if err: st.error(f"Failed #{i+1}: {err}")
                    st.session_state.retouch_results = rt_temp_results
                    st.success("Batch Complete!"); st.rerun()

    if st.session_state.retouch_results:
        st.divider(); st.subheader("🎨 Retouched Results")
        try:
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for i, res_bytes in enumerate(st.session_state.retouch_results):
                    if res_bytes: zf.writestr(f"retouched_{i+1}.jpg", res_bytes)
            st.download_button("📦 Download All (.zip)", data=zip_buf.getvalue(), file_name="retouched.zip", mime="application/zip", type="primary", key=f"rt_dl_all_{rt_key_id}")
        except: pass
        cols = st.columns(3)
        for i, res_bytes in enumerate(st.session_state.retouch_results):
            with cols[i%3]:
                st.write(f"**#{i+1}**")
                if res_bytes: st.image(res_bytes, use_column_width=True)
                else: st.error("Failed")
        st.markdown("---"); st.subheader("🚀 Upload to Shopify")
        with st.container(border=True):
            rt_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            rt_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            current_imp_id = st.session_state.get(f"rt_imp_id_{rt_key_id}", "")
            col_rt_u1, col_rt_u2 = st.columns([2, 1])
            rt_prod_id = col_rt_u1.text_input("Product ID", value=current_imp_id, key=f"rt_upload_id_{rt_key_id}")
            if col_rt_u2.button("☁️ Upload & Replace", type="primary", use_container_width=True, key=f"rt_upload_btn_{rt_key_id}"):
                if not rt_shop or not rt_token: st.error("Missing Secrets")
                elif not rt_prod_id: st.warning("Enter ID")
                elif not any(st.session_state.retouch_results): st.warning("No images")
                else:
                    with st.spinner("Uploading..."):
                        success, msg = upload_only_images_to_shopify(rt_shop, rt_token, rt_prod_id, st.session_state.retouch_results)
                        if success: st.success(msg); st.balloons()
                        else: st.error(msg)

    st.markdown("---"); st.subheader("🛍️ SEO Name & Slug Generator")
    current_text_model = st.session_state.get('selected_text_model', 'Gemini')
    st.caption(f"🤖 Using: **{current_text_model}**")
    target_images_for_seo = []
    if st.session_state.retouch_results and any(st.session_state.retouch_results):
        target_images_for_seo = [x for x in st.session_state.retouch_results if x]
        source_label = "Retouched"
    elif rt_imgs: target_images_for_seo = rt_imgs; source_label = source_type
    else: source_label = "None"
    c_seo1, c_seo2 = st.columns([1, 1])
    with c_seo1:
        user_product_desc = st.text_input("Description", placeholder="e.g., sterling silver bracelet", key=f"rt_seo_desc_{rt_key_id}")
        st.write(f"Source: {source_label}")
        if st.button("✨ Analyze", key=f"rt_seo_analyze_btn_{rt_key_id}"):
            if not target_images_for_seo: st.warning("No images.")
            elif not user_product_desc: st.warning("Enter description.")
            else:
                with st.spinner("Analyzing..."):
                    seo_json, seo_err = generate_seo_name_slug(gemini_key, claude_key, current_text_model, target_images_for_seo, user_product_desc)
                    if seo_json:
                        res_dict = parse_json_response(seo_json)
                        if res_dict: st.session_state.seo_name_result = res_dict
                        else: st.error("Parse failed"); st.code(seo_json)
                    else: st.error(seo_err)
    with c_seo2:
        if st.session_state.seo_name_result:
            res = st.session_state.seo_name_result
            st.success("Done!")
            st.write("**Product Name:**"); st.text_input("Name", value=res.get("product_name", ""), label_visibility="collapsed", key=f"rt_res_name_{rt_key_id}")
            st.write("**URL Slug:**"); st.code(res.get("url_slug", ""))

# === TAB 2: BULK SEO ===
with tab2:
    st.header("🏷️ Bulk SEO Tags")
    current_text_model = st.session_state.get('selected_text_model', 'Gemini')
    st.caption(f"🤖 Using: **{current_text_model}**")
    bulk_key_id = st.session_state.bulk_key_counter
    bc1, bc2 = st.columns([1, 1.5])
    with bc1:
        bfiles = st.file_uploader("Upload Images", accept_multiple_files=True, key=f"bulk_up_{bulk_key_id}")
        bimgs = [Image.open(f) for f in bfiles] if bfiles else []
        if bimgs: st.success(f"{len(bimgs)} images")
    with bc2:
        burl = st.text_input("Product URL:", key=f"bulk_url_{bulk_key_id}")
        c_btn1, c_btn2 = st.columns([1, 1])
        run_batch = c_btn1.button("🚀 Run Batch", type="primary", disabled=(not bimgs), key=f"bulk_run_btn_{bulk_key_id}")
        if c_btn2.button("🔄 Start Over", key=f"bulk_startover_btn_{bulk_key_id}"):
            st.session_state.bulk_results = None; st.session_state.bulk_key_counter += 1; st.rerun()
        if run_batch:
            if (current_text_model == "Gemini" and not gemini_key) or (current_text_model == "Claude" and not claude_key): st.error("Missing API Key")
            elif not burl: st.error("Missing URL")
            else:
                pbar = st.progress(0); temp_results = []
                for i, img in enumerate(bimgs):
                    with st.spinner(f"Processing #{i+1}..."):
                        txt, err = generate_seo_for_existing_image(gemini_key, claude_key, current_text_model, img, burl)
                        pbar.progress((i+1)/len(bimgs))
                        if txt:
                            d = parse_json_response(txt)
                            if isinstance(d, list) and d: d = d[0]
                            temp_results.append(d if isinstance(d, dict) else {"error": "Invalid format", "raw": txt})
                        else: temp_results.append({"error": err})
                st.session_state.bulk_results = temp_results; st.success("Done!"); st.rerun()
    if st.session_state.bulk_results and bimgs:
        st.divider()
        for i, res in enumerate(st.session_state.bulk_results):
            if i < len(bimgs):
                rc1, rc2 = st.columns([1, 3])
                with rc1: st.image(bimgs[i], width=150)
                with rc2:
                    if "error" in res: st.error(res.get('error')); st.code(res.get('raw', '')) if 'raw' in res else None
                    else: st.write("**File Name:**"); st.code(res.get('file_name', '')); st.write("**Alt Tag:**"); st.code(res.get('alt_tag', ''))
                st.divider()

# === TAB 3: WRITER ===
with tab3:
    st.header("📝 Product Writer")
    current_text_model = st.session_state.get('selected_text_model', 'Gemini')
    st.caption(f"🤖 Using: **{current_text_model}**")
    writer_key_id = st.session_state.writer_key_counter
    if "writer_shopify_imgs" not in st.session_state: st.session_state.writer_shopify_imgs = []
    text_area_key = f"w_raw_{writer_key_id}"
    c1, c2 = st.columns([1, 1.2])
    with c1:
        with st.expander("🛍️ Import from Shopify", expanded=True):
            sh_secret_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            sh_secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            if sh_secret_shop and sh_secret_token:
                sh_writer_id = st.text_input("Product ID", key=f"writer_shopify_id_{writer_key_id}")
                col_w_fetch, col_w_clear = st.columns([2, 1])
                if col_w_fetch.button("⬇️ Fetch All", key=f"writer_fetch_btn_{writer_key_id}"):
                    if not sh_writer_id: st.warning("Enter ID")
                    else:
                        with st.spinner("Fetching..."):
                            imgs, _ = get_shopify_product_images(sh_secret_shop, sh_secret_token, sh_writer_id)
                            desc_html, _, _, _ = get_shopify_product_details(sh_secret_shop, sh_secret_token, sh_writer_id)
                            if imgs: st.session_state.writer_shopify_imgs = imgs
                            if desc_html: st.session_state[text_area_key] = remove_html_tags(desc_html)
                            st.success("Loaded!"); st.rerun()
                if col_w_clear.button("❌ Clear", key=f"writer_clear_btn_{writer_key_id}"):
                    st.session_state.writer_shopify_imgs = []
                    if text_area_key in st.session_state: st.session_state[text_area_key] = ""
                    st.rerun()
        writer_imgs = st.session_state.writer_shopify_imgs if st.session_state.writer_shopify_imgs else []
        if not writer_imgs:
            files = st.file_uploader("Images (Optional)", type=["jpg", "png"], accept_multiple_files=True, key=f"w_img_{writer_key_id}")
            writer_imgs = [Image.open(f) for f in files] if files else []
        if writer_imgs:
            with st.expander(f"📸 Preview ({len(writer_imgs)} images)", expanded=False):
                cols = st.columns(4)
                for i, img in enumerate(writer_imgs): cols[i%4].image(img, use_column_width=True)
        raw = st.text_area("Paste Details:", height=300, key=text_area_key)
        wb1, wb2 = st.columns([1, 1])
        run_write = wb1.button("🚀 Generate Content", type="primary", key=f"writer_run_btn_{writer_key_id}")
        if wb2.button("🔄 Start Over", key=f"writer_startover_btn_{writer_key_id}"):
            st.session_state.writer_result = None; st.session_state.writer_shopify_imgs = []; st.session_state.writer_key_counter += 1; st.rerun()
    with c2:
        if run_write:
            if (current_text_model == "Gemini" and not gemini_key) or (current_text_model == "Claude" and not claude_key): st.error("Missing API Key")
            elif not raw: st.error("Missing details")
            else:
                with st.spinner(f"Writing with {current_text_model}..."):
                    json_txt, err = generate_full_product_content(gemini_key, claude_key, current_text_model, writer_imgs, raw)
                    if json_txt:
                        d = parse_json_response(json_txt)
                        if isinstance(d, list) and d: d = d[0]
                        if isinstance(d, dict): st.session_state.writer_result = d; st.rerun()
                        else: st.code(json_txt)
                    else: st.error(err)
        if st.session_state.writer_result:
            d = st.session_state.writer_result
            st.subheader("Content Results")
            st.write("**H1:**"); st.code(d.get('product_title_h1', ''))
            st.write("**Slug:**"); st.code(d.get('url_slug', ''))
            st.write("**Meta Title:**"); st.code(d.get('meta_title', ''))
            st.write("**Meta Desc:**"); st.code(d.get('meta_description', ''))
            with st.expander("HTML Content"): st.code(d.get('html_content', ''), language="html")
            st.markdown(d.get('html_content', ''), unsafe_allow_html=True)
            st.divider(); st.subheader("🖼️ Image SEO")
            img_tags = d.get('image_seo', [])
            if writer_imgs:
                for i, img in enumerate(writer_imgs):
                    ic1, ic2 = st.columns([1, 3])
                    with ic1: st.image(img, width=120)
                    with ic2:
                        if i < len(img_tags):
                            item = img_tags[i]
                            st.write("**File:**"); st.code(clean_filename(item.get('file_name', '')) if isinstance(item, dict) else "N/A")
                            st.write("**Alt:**"); st.code(item.get('alt_tag', '') if isinstance(item, dict) else str(item))
                    st.divider()
            st.markdown("---"); st.subheader("🚀 Publish to Shopify")
            with st.container(border=True):
                secret_shop = st.secrets.get("SHOPIFY_SHOP_URL")
                secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN")
                s_shop, s_token, s_prod_id = None, None, None
                if secret_shop and secret_token:
                    col_info, col_input = st.columns([1, 1])
                    with col_info: st.success("✅ Credentials Loaded"); s_shop = secret_shop; s_token = secret_token
                    with col_input: s_prod_id = st.text_input("Product ID", value=st.session_state.get(f"writer_shopify_id_{writer_key_id}", ""), key=f"writer_prod_id_{writer_key_id}")
                else: st.warning("⚠️ Credentials Required"); c_x1, c_x2, c_x3 = st.columns(3); s_shop = c_x1.text_input("Shop URL", key=f"writer_shop_{writer_key_id}"); s_token = c_x2.text_input("Token", type="password", key=f"writer_token_{writer_key_id}"); s_prod_id = c_x3.text_input("Product ID", key=f"writer_prodid2_{writer_key_id}")
                enable_img_upload = st.checkbox("📷 Upload Images", value=True, key=f"writer_imgchk_{writer_key_id}")
                if st.button("☁️ Update Product", type="primary", use_container_width=True, key=f"writer_update_btn_{writer_key_id}"):
                    if not s_shop or not s_token or not s_prod_id: st.error("❌ Missing Data")
                    else:
                        with st.spinner("Updating..."):
                            success, msg = update_shopify_product_v2(s_shop, s_token, s_prod_id, st.session_state.writer_result, writer_imgs, enable_img_upload)
                            if success: st.success(msg); st.balloons()
                            else: st.error(msg)

# === TAB 4: LIBRARY ===
with tab4:
    st.subheader("🛠️ Library Manager")
    target = st.session_state.edit_target
    
    # Use dynamic form key based on whether editing or adding
    form_key = f"lib_form_{target['id']}" if target else "lib_form_new"
    
    with st.form(form_key, clear_on_submit=True):
        st.write(f"**{'Edit: '+target['name'] if target else 'Add New'}**")
        c1, c2 = st.columns(2)
        
        # Use dynamic keys with target id to force refresh
        key_suffix = target['id'] if target else "new"
        n = c1.text_input("Name", value=target['name'] if target else "", key=f"lib_name_{key_suffix}")
        c = c2.text_input("Category", value=target['category'] if target else "", key=f"lib_cat_{key_suffix}")
        t = st.text_area("Template", value=target['template'] if target else "", key=f"lib_template_{key_suffix}")
        v = st.text_input("Vars (comma separated)", value=target['variables'] if target else "", key=f"lib_vars_{key_suffix}")
        u = st.text_input("Sample URL", value=target['sample_url'] if target else "", key=f"lib_url_{key_suffix}")
        
        cols = st.columns([1, 1, 3])
        save_btn = cols[0].form_submit_button("💾 Save", type="primary")
        cancel_btn = cols[1].form_submit_button("❌ Cancel") if target else False
        
        if save_btn:
            new = {"id": target['id'] if target else str(int(time.time())), "name": n, "category": c, "template": t, "variables": v, "sample_url": u}
            if target:
                for idx, item in enumerate(st.session_state.library):
                    if item['id'] == target['id']: st.session_state.library[idx] = new; break
            else: 
                st.session_state.library.append(new)
            save_prompts(st.session_state.library)
            st.session_state.edit_target = None
            st.success("✅ Saved!")
            st.rerun()
            
        if cancel_btn: 
            st.session_state.edit_target = None
            st.rerun()
    
    st.divider()
    st.write("**📚 Prompt Library:**")
    for i, p in enumerate(st.session_state.library):
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([1, 4, 1, 1])
            if p.get("sample_url"):
                with c1: safe_st_image(p["sample_url"], width=50)
            else:
                c1.write("📝")
            c2.write(f"**{p.get('name')}** ({p.get('category', 'N/A')})")
            if c3.button("✏️ Edit", key=f"lib_edit_{i}"): 
                st.session_state.edit_target = p
                st.rerun()
            if c4.button("🗑️ Del", key=f"lib_del_{i}"): 
                st.session_state.library.pop(i)
                save_prompts(st.session_state.library)
                st.rerun()

# === TAB 5: MODELS ===
with tab5:
    st.subheader("📊 Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Current Configuration:**")
        st.json({"Image Generation": "Gemini (gemini-3-pro-image-preview)", "Text/SEO Model": st.session_state.get('selected_text_model', 'Gemini'), "Claude Model": MODEL_TEXT_CLAUDE, "Gemini Text Model": MODEL_TEXT_GEMINI})
    with col2:
        st.write("**API Status:**")
        if gemini_key: st.success("✅ Gemini API Key: Configured")
        else: st.error("❌ Gemini API Key: Missing")
        if claude_key: st.success("✅ Claude API Key: Configured")
        else: st.warning("⚠️ Claude API Key: Not Set")
    st.divider()
    if st.button("📡 Scan Gemini Models", key="models_scan_btn"):
        if not gemini_key: st.error("No Key")
        else:
            with st.spinner("Scanning..."):
                m = list_available_models(gemini_key)
                if m:
                    gem = [x for x in m if "gemini" in x['name']]
                    st.success(f"Found {len(gem)} models")
                    st.dataframe(pd.DataFrame(gem)[['name','version','displayName']], use_container_width=True)
                else: st.error("Failed")
