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
st.set_page_config(layout="wide", page_title="Jewelry AI Studio Pro")

# Model IDs
MODEL_IMAGE_GEN = "models/gemini-3-pro-image-preview"
MODEL_TEXT_SEO_GEMINI = "models/gemini-3-pro-preview"
MODEL_TEXT_SEO_CLAUDE = "claude-sonnet-4-20250514"

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
คุณมีหน้าที่เป็นผู้เชี่ยวชาญ SEO specialist product content writer ผู้มีประสบการ์ 15-20 ปี ช่วยเขียน SEO-Optimized product description เป็นภาษาอังกฤษสำหรับร้าน e-commerce ของฉันที่สร้างโดย Shopify ตามโครงสร้าง <h1><h2> with human-written style that pass AI Content detector app https://undetectable.ai 

เป้าหมายเพื่อเพิ่มอันดับบน organic search engine และ AI search แนะนำ product ของฉันให้กับลูกค้า และเพื่อให้ได้ติดอันดับที่ดีบน organic search engine โดยมีกลุ่มเป้าหมายคือผู้ที่ค้นหา product นั้นๆจาก organic search engine ต่างๆ รวมทั้ง AI search ช่วยเขียนในภาษาที่อ่านง่ายสไตล์ Fact-driven และเข้าใจง่ายเพื่อดึงดูดลูกค้าให้ตัดสินใจซื้อได้ง่าย

ให้คุณแบ่งการวางคีย์เวิร์ดตามโครงสร้างของ Product Description ดังนี้:

1. ย่อหน้าแรก (Opening Paragraph) - บอก Google และผู้ใช้ให้ชัดเจนที่สุดว่าหน้านี้เกี่ยวกับอะไร
2. ส่วนกลางของเนื้อหา (Body of the Content) - เล่าเรื่องราว, อธิบายดีไซน์, บอกคุณสมบัติ
3. ส่วนคุณสมบัติ (Specifications / Beautiful Bullet Points) - ให้ข้อมูลทางเทคนิคที่ชัดเจน และช่วยทำตัวหนาและใส่สี font ข้อมูลที่เป็นตัวเลข เช่น 110 grams, 16 mm
4. ส่วนคำถามที่พบบ่อย (FAQ Section) - ตอบข้อสงสัยและให้ข้อมูลเพิ่มเติม

โครงสร้าง:
- Product Overview
- Key Features at a Glance / Key Features & Benefits / What Makes This Special (เลือกตามเหมาะสม)
- Frequently Asked Questions (FAQ)

รวมทั้งช่วยเขียน Google SEO-optimized meta title (approximately 60 characters), meta description, SEO-optimized image file name พร้อมกับ image alt tag สำหรับ product นี้สำหรับทุก images และ url slug

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

SEO_PROMPT_NAME_SLUG = """
You are an SEO expert with 10-15 years of experience. 
Your task is to analyze the provided product images and the user's initial description. 
Please generate:
1. An attractive, SEO-optimized Product Name.
2. A suitable, clean URL Slug (using hyphens).

User Input Description: "{user_desc}"

IMPORTANT: You MUST return the result in RAW JSON format ONLY (no markdown backticks).
Structure:
{
  "product_name": "Sterling Silver Charm Bracelet - Handcrafted",
  "url_slug": "sterling-silver-charm-bracelet-handcrafted"
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
        "id": "rt1", "name": "Clean Studio Look", "category": "Retouch",
        "template": "Retouch this jewelry product to have a clean white studio background. Enhance the metal shine of {metal_type} and gemstone clarity. Professional product photography.",
        "variables": "metal_type",
        "sample_url": ""
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

# --- CLAUDE API FUNCTION ---
def call_claude_api(api_key, prompt, images_base64_list=None, max_tokens=4000):
    """เรียก Claude API สำหรับงานเขียน SEO"""
    key = clean_key(api_key)
    url = "https://api.anthropic.com/v1/messages"
    
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    content = []
    
    if images_base64_list:
        for img_b64 in images_base64_list:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64
                }
            })
    
    content.append({
        "type": "text",
        "text": prompt
    })
    
    payload = {
        "model": MODEL_TEXT_SEO_CLAUDE,
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": content
        }]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            result = response.json()
            text_content = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    text_content += block.get("text", "")
            return text_content, None
        else:
            return None, f"Claude API Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"Request failed: {str(e)}"

# --- UNIFIED TEXT GENERATION (เลือก Model) ---
def generate_text_content(selected_model, gemini_key, claude_key, prompt, images_pil=None):
    """ฟังก์ชันกลางสำหรับเรียก AI ตามที่เลือก"""
    
    if selected_model == "Claude":
        images_b64 = []
        if images_pil:
            for img in images_pil:
                images_b64.append(img_to_base64(img))
        
        return call_claude_api(claude_key, prompt, images_b64)
    
    else:  # Gemini
        key = clean_key(gemini_key)
        url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_TEXT_SEO_GEMINI}:generateContent?key={key}"
        
        parts = [{"text": prompt}]
        if images_pil:
            for img in images_pil:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_to_base64(img)
                    }
                })
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.7,
                "responseMimeType": "application/json"
            }
        }
        
        for attempt in range(3):
            try:
                res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
                if res.status_code == 200:
                    content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                    return content.get("text"), None
                elif res.status_code == 503:
                    time.sleep(3)
                    continue
                else:
                    return None, f"Error {res.status_code}: {res.text}"
            except Exception as e:
                time.sleep(1)
        return None, "Failed after 3 attempts"

# --- SHOPIFY HELPER FUNCTIONS ---
def update_shopify_product_v2(shop_url, access_token, product_id, data, images_pil=None, upload_images=False):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"):
        shop_url += ".myshopify.com"
        
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json"
    }
    
    product_payload = {
        "id": product_id,
        "title": data.get('product_title_h1'),
        "body_html": data.get('html_content'),
        "metafields": [
            {
                "namespace": "global",
                "key": "title_tag",
                "value": data.get('meta_title', ''),
                "type": "single_line_text_field"
            },
            {
                "namespace": "global",
                "key": "description_tag",
                "value": data.get('meta_description', ''),
                "type": "multi_line_text_field"
            }
        ]
    }
    
    if upload_images and images_pil and "image_seo" in data:
        img_payloads = []
        image_seo_list = data.get("image_seo", [])
        
        for i, img in enumerate(images_pil):
            seo_info = image_seo_list[i] if i < len(image_seo_list) else {}
            b64_str = img_to_base64(img)
            img_entry = {
                "attachment": b64_str,
                "filename": seo_info.get("file_name", f"image_{i+1}.jpg"),
                "alt": seo_info.get("alt_tag", "")
            }
            img_payloads.append(img_entry)
            
        if img_payloads:
            product_payload["images"] = img_payloads

    try:
        response = requests.put(url, json={"product": product_payload}, headers=headers)
        
        if response.status_code in [200, 201]:
            return True, "✅ Update Successful!"
        else:
            return False, f"Shopify API Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def add_single_image_to_shopify(shop_url, access_token, product_id, image_bytes, file_name=None, alt_tag=None):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"):
        shop_url += ".myshopify.com"
        
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}/images.json"
    
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json"
    }
    
    if not image_bytes:
        return False, "No valid image data."

    b64_str = base64.b64encode(image_bytes).decode('utf-8')
    
    final_filename = file_name if file_name else f"gen_ai_image_{int(time.time())}.jpg"
    final_alt = alt_tag if alt_tag else "AI Generated Product Image"

    payload = {
        "image": {
            "attachment": b64_str,
            "filename": final_filename, 
            "alt": final_alt
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code in [200, 201]:
            return True, "✅ Added Successful!"
        else:
            return False, f"Shopify Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def upload_only_images_to_shopify(shop_url, access_token, product_id, image_bytes_list):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"):
        shop_url += ".myshopify.com"
        
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json"
    }
    
    img_payloads = []
    for i, img_bytes in enumerate(image_bytes_list):
        if img_bytes:
            b64_str = base64.b64encode(img_bytes).decode('utf-8')
            img_payloads.append({
                "attachment": b64_str,
                "filename": f"retouched_image_{i+1}.jpg",
                "alt": f"Retouched Product Image {i+1}"
            })
            
    if not img_payloads:
        return False, "No valid images to upload."

    payload = {
        "product": {
            "id": product_id,
            "images": img_payloads
        }
    }
    
    try:
        response = requests.put(url, json=payload, headers=headers)
        if response.status_code in [200, 201]:
            return True, "✅ Upload Successful!"
        else:
            return False, f"Shopify Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"
        
def clean_filename(name):
    if not name: return "N/A"
    clean = re.sub(r'[^a-zA-Z0-9\-\_\.]', '', str(name))
    return clean.rsplit('.', 1)[0]

def get_shopify_product_images(shop_url, access_token, product_id):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"):
        shop_url += ".myshopify.com"
        
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}/images.json"
    
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            images_data = data.get("images", [])
            
            pil_images = []
            for img_info in images_data:
                src = img_info.get("src")
                if src:
                    img_resp = requests.get(src, stream=True)
                    if img_resp.status_code == 200:
                        img_pil = Image.open(BytesIO(img_resp.content))
                        if img_pil.mode in ('RGBA', 'P'):
                            img_pil = img_pil.convert('RGB')
                        pil_images.append(img_pil)
            return pil_images, None
        else:
            return None, f"Shopify API Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

def get_shopify_product_details(shop_url, access_token, product_id):
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"):
        shop_url += ".myshopify.com"
        
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    headers = { "X-Shopify-Access-Token": access_token, "Content-Type": "application/json" }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            prod = response.json().get("product", {})
            return prod.get("body_html", ""), prod.get("title", ""), prod.get("handle", ""), None
        else:
            return None, None, None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, None, None, str(e)

def remove_html_tags(text):
    if not text: return ""
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<')
    return "\n".join([line.strip() for line in text.split('\n') if line.strip()])

# --- AI FUNCTIONS (GEMINI - สำหรับ Image Gen เท่านั้น) ---
def generate_image(api_key, image_list, prompt):
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_IMAGE_GEN}:generateContent?key={key}"
    
    full_prompt = f"Instruction: {prompt} \nImportant Constraint: Keep the main jewelry product in the input image EXACTLY as it looks (same shape, design, texture). Only improve the lighting, background, and overall photography quality. Do not hallucinate new details on the product itself."
    
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

def list_available_models(api_key):
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200: return response.json().get("models", [])
        return None
    except: return None

# --- SESSION STATE INIT ---
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

# --- SIDEBAR: API KEYS & MODEL SELECTION ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Gemini API Key
    if "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini Key Loaded")
    elif "GOOGLE_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ Google Key Loaded")
    else:
        gemini_api_key = st.text_input("Gemini API Key", type="password")
    gemini_api_key = clean_key(gemini_api_key)
    
    # Claude API Key
    if "CLAUDE_API_KEY" in st.secrets:
        claude_api_key = st.secrets["CLAUDE_API_KEY"]
        st.success("✅ Claude Key Loaded")
    else:
        claude_api_key = st.text_input("Claude API Key (Optional)", type="password", help="For advanced SEO writing")
    claude_api_key = clean_key(claude_api_key)

    # Model Selection
    st.divider()
    st.subheader("🤖 AI Model Selection")
    
    available_models = []
    if gemini_api_key:
        available_models.append("Gemini 3 Pro")
    if claude_api_key:
        available_models.append("Claude Sonnet 4")
    
    if available_models:
        selected_text_model = st.selectbox(
            "SEO Writing Model:",
            available_models,
            help="เลือก AI Model สำหรับงานเขียน SEO (Image Gen ยังใช้ Gemini อยู่)"
        )
    else:
        st.warning("⚠️ No AI model available")
        selected_text_model = None
    
    st.caption(f"📸 Image Generation: **Gemini 3 Pro** (Fixed)")
    
    if "JSONBIN_API_KEY" in st.secrets:
        st.caption("✅ Database Connected")
    else:
        st.warning("⚠️ Local Mode")

st.title("💎 Jewelry AI Studio Pro")
tab1, tab_retouch, tab2, tab3, tab4, tab5 = st.tabs(["✨ Gen Image", "🎨 Retouch", "🏷️ Bulk SEO", "📝 Writer", "📚 Library", "ℹ
                                                     # === TAB 3: WRITER (UPDATED WITH MODEL SELECTION) ===
with tab3:
    st.header("📝 Product Writer")
    writer_key_id = st.session_state.writer_key_counter
    
    if "writer_shopify_imgs" not in st.session_state: st.session_state.writer_shopify_imgs = []
    
    text_area_key = f"w_raw_{writer_key_id}"
    
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        with st.expander("🛍️ Import from Shopify", expanded=True):
            sh_secret_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            sh_secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            
            if sh_secret_shop and sh_secret_token:
                sh_writer_id = st.text_input("Product ID", key="writer_shopify_id")
                
                col_w_fetch, col_w_clear = st.columns([2, 1])
                
                if col_w_fetch.button("⬇️ Fetch All", key="writer_fetch_btn"):
                    if not sh_writer_id: st.warning("Enter ID")
                    else:
                        with st.spinner("Fetching Data..."):
                            imgs, err_img = get_shopify_product_images(sh_secret_shop, sh_secret_token, sh_writer_id)
                            desc_html, title, _, err_desc = get_shopify_product_details(sh_secret_shop, sh_secret_token, sh_writer_id)
                            
                            if imgs:
                                st.session_state.writer_shopify_imgs = imgs
                            
                            if desc_html is not None: 
                                clean_desc = remove_html_tags(desc_html)
                                st.session_state[text_area_key] = clean_desc
                                
                            st.success("Loaded!")
                            st.rerun()
                            
                if col_w_clear.button("❌ Clear", key="writer_clear_btn"):
                    st.session_state.writer_shopify_imgs = []
                    if text_area_key in st.session_state:
                        st.session_state[text_area_key] = ""
                    st.rerun()
                    
        writer_imgs = []
        if st.session_state.writer_shopify_imgs:
            writer_imgs = st.session_state.writer_shopify_imgs
            st.info(f"Using {len(writer_imgs)} images from Shopify")
        else:
            files = st.file_uploader("Images (Optional)", type=["jpg", "png"], accept_multiple_files=True, key=f"w_img_{writer_key_id}")
            writer_imgs = [Image.open(f) for f in files] if files else []
        
        if writer_imgs:
            with st.expander("📸 Image Preview", expanded=False):
                cols = st.columns(4)
                for i, img in enumerate(writer_imgs):
                    cols[i%4].image(img, use_column_width=True, caption=f"#{i+1}")

        raw = st.text_area("Paste Details:", height=300, key=text_area_key)
        
        wb1, wb2 = st.columns([1, 1])
        run_write = wb1.button("🚀 Generate Content", type="primary")
        clear_write = wb2.button("🔄 Start Over", key="clear_writer")
        
        if clear_write:
            st.session_state.writer_result = None
            st.session_state.writer_shopify_imgs = []
            st.session_state.writer_key_counter += 1
            st.rerun()

    with c2:
        if run_write:
            if not raw:
                st.error("Missing Info")
            else:
                # เลือก Model ที่จะใช้
                if selected_text_model == "Claude Sonnet 4":
                    if not claude_api_key:
                        st.error("Claude API Key required!")
                    else:
                        with st.spinner("🤖 Claude is writing your content..."):
                            prompt = SEO_PRODUCT_WRITER_PROMPT.replace("{raw_input}", raw)
                            json_txt, err = generate_text_content(
                                "Claude",
                                gemini_api_key,
                                claude_api_key,
                                prompt,
                                writer_imgs
                            )
                            if json_txt:
                                d = parse_json_response(json_txt)
                                if isinstance(d, list) and len(d) > 0: d = d[0]
                                if isinstance(d, dict):
                                    st.session_state.writer_result = d
                                    st.success("✅ Content generated by Claude!")
                                    st.rerun()
                                else: 
                                    st.error("Invalid JSON format")
                                    st.code(json_txt)
                            else: 
                                st.error(err)
                else:  # Gemini
                    if not gemini_api_key:
                        st.error("Gemini API Key required!")
                    else:
                        with st.spinner("🤖 Gemini is writing your content..."):
                            prompt = SEO_PRODUCT_WRITER_PROMPT.replace("{raw_input}", raw)
                            json_txt, err = generate_text_content(
                                "Gemini",
                                gemini_api_key,
                                claude_api_key,
                                prompt,
                                writer_imgs
                            )
                            if json_txt:
                                d = parse_json_response(json_txt)
                                if isinstance(d, list) and len(d) > 0: d = d[0]
                                if isinstance(d, dict):
                                    st.session_state.writer_result = d
                                    st.success("✅ Content generated by Gemini!")
                                    st.rerun()
                                else: st.code(json_txt)
                            else: st.error(err)

        if st.session_state.writer_result:
            d = st.session_state.writer_result
            st.subheader("Content Results")
            st.write("Product Title (H1):"); st.code(d.get('product_title_h1', ''), language="text")
            st.write("Slug Handle:"); st.code(d.get('url_slug', ''), language="text")
            st.write("Meta Title:"); st.code(d.get('meta_title', ''), language="text")
            st.write("Meta Description:"); st.code(d.get('meta_description', ''), language="text")
            
            with st.expander("HTML Content"): st.code(d.get('html_content', ''), language="html")
            st.markdown(d.get('html_content', ''), unsafe_allow_html=True)
            
            st.divider()
            st.subheader("🖼️ Image SEO Mapping")
            
            img_tags = d.get('image_seo', [])
            
            if writer_imgs:
                for i, img in enumerate(writer_imgs):
                    with st.container():
                        ic1, ic2 = st.columns([1, 3])
                        with ic1:
                            st.image(img, width=120, caption=f"Img #{i+1}")
                        with ic2:
                            if i < len(img_tags):
                                item = img_tags[i]
                                fname = clean_filename(item.get('file_name', 'N/A')) if isinstance(item, dict) else "N/A"
                                atag = item.get('alt_tag', 'N/A') if isinstance(item, dict) else str(item)
                                st.write("**File Name:**"); st.code(fname, language="text")
                                st.write("**Alt Tag:**"); st.code(atag, language="text")
                        st.divider()

            st.markdown("---")
            st.subheader("🚀 Automation: Publish to Shopify")
            
            with st.container(border=True):
                st.info("ℹ️ ระบบจะอัปเดต: Title, Description, Meta Tags และรูปภาพ (ถ้าเลือก)")
                
                secret_shop = st.secrets.get("SHOPIFY_SHOP_URL")
                secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN")
                
                if secret_shop and secret_token:
                    col_info, col_input = st.columns([1, 1])
                    with col_info:
                        st.success("✅ Credentials Loaded")
                    with col_input:
                        default_id = st.session_state.get("writer_shopify_id", "")
                        s_prod_id = st.text_input("Product ID", value=default_id)
                    s_shop = secret_shop
                    s_token = secret_token
                else:
                    st.warning("⚠️ Credentials Required")
                    c_x1, c_x2, c_x3 = st.columns(3)
                    s_shop = c_x1.text_input("Shop URL")
                    s_token = c_x2.text_input("Token", type="password")
                    s_prod_id = c_x3.text_input("Product ID")

                enable_img_upload = st.checkbox("📷 Upload Images & Replace Existing", value=True)
                
                if st.button("☁️ Update Product to Shopify Now", type="primary", use_container_width=True):
                    if not s_shop or not s_token or not s_prod_id:
                        st.error("❌ Missing Data")
                    else:
                        with st.spinner("Updating..."):
                            success, msg = update_shopify_product_v2(
                                shop_url=s_shop,
                                access_token=s_token,
                                product_id=s_prod_id,
                                data=st.session_state.writer_result,
                                images_pil=writer_imgs,
                                upload_images=enable_img_upload
                            )
                            if success: st.success(msg); st.balloons()
                            else: st.error(msg)

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
    st.header("ℹ️ Available Models")
    st.info("📸 **Image Generation:** Fixed on Gemini 3 Pro Image Preview")
    st.info(f"📝 **SEO Writing:** Currently using **{selected_text_model if selected_text_model else 'None'}**")
    
    if st.button("📡 Scan Gemini Models"):
        if not gemini_api_key: st.error("No Gemini API Key")
        else:
            with st.spinner("Scanning..."):
                m = list_available_models(gemini_api_key)
                if m:
                    gem = [x for x in m if "gemini" in x['name']]
                    st.success(f"Found {len(gem)} Gemini models")
                    st.dataframe(pd.DataFrame(gem)[['name','version','displayName']], use_container_width=True)
                else: st.error("Failed to fetch models")
