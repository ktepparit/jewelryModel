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

# Claude Models
CLAUDE_MODELS = {
    "Claude Sonnet 4.5": "claude-sonnet-4-5-20250929",
    "Claude Opus 4.6": "claude-opus-4-6",
}

# OpenAI Models (Chat Completions API compatible)
OPENAI_MODELS = {
    "GPT-5.2": "gpt-5.2",
}

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

SEO_PROMPT_IMAGE_ANALYSIS = """
You are an SEO & Visual Content Specialist for Jewelry e-commerce with 15-20 years of experience.
Your task is to analyze the GENERATED IMAGE provided and create SEO-optimized **Image File Name** and **Alt Tag**.

**Product Reference URL:** "{product_url}"

**Instructions:**
1. **ANALYZE THE IMAGE** - Look at the actual generated image and describe what you see:
   - Type of jewelry (ring, necklace, bracelet, earrings, etc.)
   - Materials visible (gold, silver, platinum, etc.)
   - Gemstones (diamond, sapphire, ruby, etc.)
   - Style (modern, vintage, minimalist, luxury, etc.)
   - Visual elements (model wearing it, product shot, lifestyle, etc.)
   - Background and lighting style

2. **File Name:** Create a lowercase, hyphenated file name ending in .jpg
   - COMBINE product keywords from URL with VISUAL details from the image
   - Include: material, product type, style, and visual context
   - Example: `gold-diamond-ring-elegant-hand-model-lifestyle.jpg`

3. **Alt Tag:** Write a natural English sentence describing exactly what is shown in the image
   - Be specific about what you SEE in the image
   - Mention materials, style, and context visible
   - Good for accessibility and SEO

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
Act as an expert E-commerce Copywriter and SEO Specialist who masters Semantic SEO, Query Fan-out, and Google's E-E-A-T principles.
เป้าหมายเพื่อเพิ่มอันดับบน organic search engine และ AI search แนะนำ product ของฉันให้กับลูกค้า

**Strategic Instructions:**

1.  **Query Fan-out Strategy:** Identify 3 distinct user sub-intents (e.g., specific problems, usage occasions, or user types) and address them specifically in the body content.
2.  **Demonstrate "Experience" (E-E-A-T):** Write as if you have physically tested the product. Include *sensory details* to prove experience (e.g., describe the texture, the sound of the click, the weight in hand, or the setup process). Avoid generic fluff.
3.  **Semantic & NLP Optimization:** Use related entities and semantically relevant topics, not just keywords. Focus on the "solution" rather than just the "spec."
4.  **Structured Data for AI (The Table):** At the end of the description, create a summary table named "Quick Specs & Real-World Performance" that maps technical specs to actual user benefits.

**Output Structure:**

* **Catchy Headline:** Benefit-driven and catchy. Focus on the main benefit/outcome
* **The "Hands-On" Intro:** Introduce the product with a focus on the feeling/experience of using it.
* **Detailed Usage Scenarios (Fan-out):** Use sub-headers for different user needs.
* **Key Specs (Contextualized):** List the specs but explain the *benefit* of each spec immediately (e.g., "30W Fast Charge: Get 50% battery in just coffee-break time").
* **Q&A Section:** Answer 3 common questions related to buying decisions.
* **Summary Table:** A 2-column table: [Feature/Spec] | [Real-World Benefit].

**Tone:** Authentic, Experienced, Helpful, and Human.


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
# --- CLAUDE API FUNCTION ---
# ============================================================
def call_claude_api(claude_key, prompt, img_pil_list=None, model_id="claude-sonnet-4-20250514"):
    """Call Claude API for Text/SEO tasks with optional image support"""
    url = "https://api.anthropic.com/v1/messages"
    headers = {"Content-Type": "application/json", "x-api-key": claude_key, "anthropic-version": "2023-06-01"}
    
    content = []
    if img_pil_list:
        for img in img_pil_list:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_to_base64(img)}})
    content.append({"type": "text", "text": prompt})
    
    payload = {"model": model_id, "max_tokens": 4096, "messages": [{"role": "user", "content": content}]}
    
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
# --- OPENAI API FUNCTION (NEW) ---
# ============================================================
def call_openai_api(openai_key, prompt, img_pil_list=None, model_id="gpt-5.2"):
    """Call OpenAI API for Text/SEO tasks with optional image support"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"}
    
    content = []
    if img_pil_list:
        for img in img_pil_list:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_to_base64(img)}"}})
    content.append({"type": "text", "text": prompt})
    
    payload = {
        "model": model_id, 
        "max_completion_tokens": 4096,  # GPT-5.2 uses max_completion_tokens instead of
