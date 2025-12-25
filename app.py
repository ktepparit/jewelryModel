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
MODEL_TEXT_SEO = "models/gemini-3-pro-preview"

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

# --- NEW PROMPT: GEN TAGS FROM PROMPT TEXT ---
SEO_PROMPT_FROM_TEXT = """
You are an SEO specialist for Jewelry e-commerce.
Based on this product image description/prompt: "{context}"
Generate:
1. An SEO-optimized image file name (lowercase, use hyphens, end with .jpg).
2. A descriptive Image Alt Tag (English).

IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure: {"file_name": "silver-ring-example.jpg", "alt_tag": "Description of the ring"}
"""

SEO_PROMPT_BULK_EXISTING = """
คุณคือ SEO specialist ที่มีประสบการณ์ 15-20 ปี ช่วยเขียน SEO-optimized image file name with image alt tags เป็นภาษาอังกฤษ สำหรับสินค้าของฉันตามแต่ละรูปที่แนบมาให้ {product_url} เพื่อให้ได้ติดอันดับที่ดีบน organic search engine โดยกลุ่มลูกค้าเป็นผู้สนใจสินค้าชนิดนี้
IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure: {"file_name": "...", "alt_tag": "..."}
"""

SEO_PRODUCT_WRITER_PROMPT = """
คุณมีหน้าที่เป็นผู้เชี่ยวชาญ SEO specialist product content writer ผู้มีประสบการ์ 15-20 ปี ช่วยเขียน SEO-Optimized product description เป็นภาษาอังกฤษสำหรับร้าน

e-commerce ของฉันที่สร้างโดยShopify ตามโครงสร้าง

<h1><h2>  with human-written style that pass AI Content

detector app https://undetectable.ai 

เป้าหมายเพื่อเพิ่มอันดับบน organic search engine และ AI

search แนะนำ product ของฉันให้กับลูกค้า และเพื่อให้ได้ติดอันดับที่ดีบน organic search engine โดยมีกลุ่มเป้าหมายคือผู้ที่ค้นหา product นั้นๆจากorganic search engine ต่างๆ รวมทั้ง AI search ช่วยเขียนในภาษาที่อ่านง่ายสไตล์ Fact-driven และเข้าใจง่ายเพื่อดึงดูดลูกค้าให้ตัดสินใจซื้อได้ง่าย

ให้คุณแบ่งการวางคีย์เวิร์ดตามโครงสร้างของ Product Description ดังนี้ครับ:

1. ย่อหน้าแรก (Opening Paragraph)

    เป้าหมาย: บอก Google และผู้ใช้ให้ชัดเจนที่สุดว่าหน้านี้เกี่ยวกับอะไร

    สิ่งที่ควรวาง:

        คีย์เวิร์ดหลัก (Primary Keyword): เน้นที่คำว่า medusa ring ให้ชัดเจน

        Semantic Keyword ที่สำคัญที่สุด 1-2 คำ: อาจจะใส่คำว่า Greek mythology หรือ Gorgon เข้าไปถ้ามันทำให้ประโยคดูสละสลวยขึ้น


2. ส่วนกลางของเนื้อหา (Body of the Content)

    เป้าหมาย: เล่าเรื่องราว, อธิบายดีไซน์, บอกคุณสมบัติ

    สิ่งที่ควรวาง: นี่คือพื้นที่ที่ดีที่สุดในการกระจาย Semantic Keywords ส่วนใหญ่

        ส่วนที่อธิบายดีไซน์: พูดถึง serpent hair (ผมที่เป็นงู) หรือ petrifying gaze (สายตาที่ทำให้เป็นหิน)

        ส่วนที่เล่าถึงที่มา/แรงบันดาลใจ: อ้างอิงถึงตำนานของ goddess Athena (เทพีอาธีน่า) ที่สาปเมดูซ่า

        ส่วนที่อธิบายสัญลักษณ์: พูดถึงความหมายของเมดูซ่าในยุคใหม่ เช่น protection, feminine power (พลังของผู้หญิง), rebellion (การขบถ)


3. ส่วนคุณสมบัติ (Specifications / Beautiful Icon Bullet Points)

    เป้าหมาย: ให้ข้อมูลทางเทคนิคที่ชัดเจน

    สิ่งที่ควรวาง: เหมาะสำหรับคำที่เกี่ยวกับวัสดุ เช่น 925 sterling silver, solid silver, handcrafted, oxidized finish


4. ส่วนคำถามที่พบบ่อย (FAQ Section)

    เป้าหมาย: ตอบข้อสงสัยและให้ข้อมูลเพิ่มเติม

    สิ่งที่ควรวาง: เป็นโอกาสที่ดีในการใช้คีย์เวิร์ดแบบยาวๆ (Long-tail keywords) และ Semantic Keywords ที่เกี่ยวข้อง

        "สัญลักษณ์ของเมดูซ่าในเครื่องประดับหมายถึงอะไร?"

        "แหวนเงินแท้ดูแลรักษายากหรือไม่?"


โดยฉันจะแจ้งรายละเอียดให้คุณดังนี้

1. url ของ product

2 คีย์เวิร์ดหลัก

3 คีย์เวิร์ดรองและ long tail keyword

4 คีย์เวิร์ดหมวดหมู่

5 รูปภาพ product (ถ้ามี)

6 รายละเอียดเพิ่มเติม (ถ้ามี)

ถ้าหากฉันไม่มีข้อมูล คีย์เวิร์ดหลัก , คีย์เวิร์ดรองและ long tail keyword และ คีย์เวิร์ดหมวดหมู่มาให้ให้คุณทำการคิดให้ฉันเลย

หลังจากนั้นให้คุณเขียน product description ตามคำสั่งข้างต้นโดยแทรก คีย์เวิร์ดรอง, คีย์เวิร์ดหมวดหมู่, Semantic Keywords และ Long-tail keywords เข้าไปยัง content ตามคำสั่งข้างต้น


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

# --- NEW PROMPT FOR NAME & SLUG GENERATOR ---
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
    # เพิ่ม Default Retouch Template เผื่อไว้ทดสอบ
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

# --- SHOPIFY HELPER FUNCTION (FULL UPDATE) ---
def update_shopify_product_v2(shop_url, access_token, product_id, data, images_pil=None, upload_images=False):
    """
    shop_url: ชื่อร้าน (subdomain) หรือ full url
    access_token: shpat_...
    product_id: ID สินค้า
    data: JSON Data จาก AI
    images_pil: List ของ PIL Images (ถ้าจะอัปโหลดรูป)
    upload_images: Boolean flag ว่าจะเอาสรูปขึ้นด้วยไหม
    """
    # Clean URL
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"):
        shop_url += ".myshopify.com"
        
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json"
    }
    
    # 1. Prepare Basic Product Data
    product_payload = {
        "id": product_id,
        "title": data.get('product_title_h1'),
        "body_html": data.get('html_content'),
        # เราไม่ใส่ "handle": data.get('url_slug') ตามที่คุณแจ้งว่าไม่ต้องการแก้ slug
        
        # SEO Metafields (Global Title & Description)
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
    
    # 2. Prepare Images (ถ้า User เลือกให้เอาขึ้น)
    if upload_images and images_pil and "image_seo" in data:
        img_payloads = []
        image_seo_list = data.get("image_seo", [])
        
        for i, img in enumerate(images_pil):
            # ดึงข้อมูล SEO ที่ตรงกับลำดับภาพ
            seo_info = image_seo_list[i] if i < len(image_seo_list) else {}
            
            # แปลงรูปเป็น Base64
            b64_str = img_to_base64(img) # ใช้ฟังก์ชันเดิมที่มีอยู่แล้ว
            
            # สร้าง Payload ของรูปภาพ
            img_entry = {
                "attachment": b64_str,
                "filename": seo_info.get("file_name", f"image_{i+1}.jpg"),
                "alt": seo_info.get("alt_tag", "")
            }
            img_payloads.append(img_entry)
            
        if img_payloads:
            # การส่ง images ไปใน PUT request จะเป็นการ Replace รูปเดิมทั้งหมด
            product_payload["images"] = img_payloads

    try:
        # ยิง Request ไป Shopify
        response = requests.put(url, json={"product": product_payload}, headers=headers)
        
        if response.status_code in [200, 201]:
            return True, "✅ Update Successful! ข้อมูล (และรูปภาพ) ถูกแก้ไขเรียบร้อยแล้ว"
        else:
            return False, f"Shopify API Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

# --- SHOPIFY HELPER: UPLOAD SINGLE IMAGE (APPEND ONLY) ---
def add_single_image_to_shopify(shop_url, access_token, product_id, image_bytes, file_name=None, alt_tag=None):
    """
    เพิ่มรูปภาพ 1 รูปเข้าไปในสินค้า (ไม่ลบรูปเก่า) - สำหรับ Gen Image Tab
    รองรับการระบุชื่อไฟล์และ Alt Tag
    """
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"):
        shop_url += ".myshopify.com"
        
    # Endpoint สำหรับเพิ่มรูป (POST .../images.json)
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}/images.json"
    
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json"
    }
    
    if not image_bytes:
        return False, "No valid image data."

    # แปลง Bytes เป็น Base64
    b64_str = base64.b64encode(image_bytes).decode('utf-8')
    
    # กำหนดชื่อไฟล์และ Alt Tag (ถ้าไม่มีให้ใช้ Default)
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
            return True, "✅ Added Successful! เพิ่มรูปภาพใหม่พร้อม SEO Tags เรียบร้อยแล้ว"
        else:
            return False, f"Shopify Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

# --- SHOPIFY HELPER: UPLOAD IMAGES (REPLACE ALL) ---
def upload_only_images_to_shopify(shop_url, access_token, product_id, image_bytes_list):
    """
    อัปโหลดรูปภาพไปแทนที่รูปเดิมทั้งหมด (Replace All) - สำหรับ Retouch Tab
    image_bytes_list: List ของข้อมูลรูปภาพแบบ Bytes (ไม่ใช่ PIL)
    """
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
            # แปลง Bytes เป็น Base64 โดยตรง
            b64_str = base64.b64encode(img_bytes).decode('utf-8')
            img_payloads.append({
                "attachment": b64_str,
                "filename": f"retouched_image_{i+1}.jpg", # ตั้งชื่อไฟล์ default
                "alt": f"Retouched Product Image {i+1}"
            })
            
    if not img_payloads:
        return False, "No valid images to upload."

    payload = {
        "product": {
            "id": product_id,
            "images": img_payloads # การส่ง key images จะเป็นการ Replace รูปเดิมทั้งหมด
        }
    }
    
    try:
        response = requests.put(url, json=payload, headers=headers)
        if response.status_code in [200, 201]:
            return True, "✅ Upload Successful! รูปภาพถูกแทนที่เรียบร้อยแล้ว"
        else:
            return False, f"Shopify Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"
        
def clean_filename(name):
    if not name: return "N/A"
    clean = re.sub(r'[^a-zA-Z0-9\-\_\.]', '', str(name))
    return clean.rsplit('.', 1)[0]

# --- SHOPIFY GET IMAGES FUNCTION ---
def get_shopify_product_images(shop_url, access_token, product_id):
    """
    ดึงรูปภาพทั้งหมดจาก Shopify Product ID
    return: List of PIL Images
    """
    # Clean URL
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
                    # Download Image Bytes
                    img_resp = requests.get(src, stream=True)
                    if img_resp.status_code == 200:
                        img_pil = Image.open(BytesIO(img_resp.content))
                        # Convert to RGB (for JPG compatibility)
                        if img_pil.mode in ('RGBA', 'P'):
                            img_pil = img_pil.convert('RGB')
                        pil_images.append(img_pil)
            return pil_images, None
        else:
            return None, f"Shopify API Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"

# --- SHOPIFY GET DETAILS FUNCTION (RETURNS 3 VALUES + HANDLE) ---
def get_shopify_product_details(shop_url, access_token, product_id):
    """
    ดึง Title, Body HTML, และ Handle ของสินค้า
    """
    shop_url = shop_url.replace("https://", "").replace("http://", "").strip()
    if not shop_url.endswith(".myshopify.com"):
        shop_url += ".myshopify.com"
        
    url = f"https://{shop_url}/admin/api/2024-01/products/{product_id}.json"
    headers = { "X-Shopify-Access-Token": access_token, "Content-Type": "application/json" }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            prod = response.json().get("product", {})
            # Return body_html, title, handle, error
            return prod.get("body_html", ""), prod.get("title", ""), prod.get("handle", ""), None
        else:
            return None, None, None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, None, None, str(e)

# (ฟังก์ชัน HTML stripper ง่ายๆ เผื่อคุณอยากแปลง HTML เป็น Text ล้วน แต่ในที่นี้จะส่ง Raw HTML ให้ก่อน)
def remove_html_tags(text):
    if not text: return ""
    # 1. แปลง <br>, </p>, </div> เป็นการขึ้นบรรทัดใหม่ เพื่อรักษาย่อหน้า
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
    
    # 2. ลบ HTML tags ทั้งหมด
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    
    # 3. แก้ไข HTML Entities พื้นฐาน (ถ้ามี)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<')
    
    # 4. ลบช่องว่างส่วนเกิน
    return "\n".join([line.strip() for line in text.split('\n') if line.strip()])


# --- AI FUNCTIONS (GEMINI) ---
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

def generate_seo_tags_from_context(api_key, context):
    """
    Gen SEO Tags based on prompt text/context (For Gen Image tab)
    """
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_TEXT_SEO}:generateContent?key={key}"
    prompt = SEO_PROMPT_FROM_TEXT.replace("{context}", context)
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
    
    num_images = len(img_pil_list) if img_pil_list else 0
    if num_images > 0:
        prompt += f"\n\nCRITICAL INSTRUCTION: You received {num_images} images. You MUST return exactly {num_images} objects in the 'image_seo' array, strictly corresponding to the order of images provided (Index 0 to {num_images-1}). Do not skip any image."

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

# --- NEW FUNCTION FOR NAME/SLUG ---
def generate_seo_name_slug(api_key, img_list, user_desc):
    key = clean_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/{MODEL_TEXT_SEO}:generateContent?key={key}"
    prompt = SEO_PROMPT_NAME_SLUG.replace("{user_desc}", user_desc)
    
    parts = [{"text": prompt}]
    # Handle both PIL Images and Bytes
    if img_list:
        for item in img_list:
            if isinstance(item, bytes):
                try:
                    img_pil = Image.open(BytesIO(item))
                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}})
                except: pass
            elif isinstance(item, Image.Image):
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(item)}})

    payload = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}}
    
    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if res.status_code == 200:
            content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
            return content.get("text"), None
        else: return None, f"Error {res.status_code}"
    except Exception as e: return None, str(e)


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
if "gen_tags_result" not in st.session_state: st.session_state.gen_tags_result = {} # Store Tags as Dict {file_name, alt_tag}

# Store results
if "bulk_results" not in st.session_state: st.session_state.bulk_results = None
if "writer_result" not in st.session_state: st.session_state.writer_result = None
if "retouch_results" not in st.session_state: st.session_state.retouch_results = None
if "seo_name_result" not in st.session_state: st.session_state.seo_name_result = None

# Widget Keys
if "bulk_key_counter" not in st.session_state: st.session_state.bulk_key_counter = 0
if "writer_key_counter" not in st.session_state: st.session_state.writer_key_counter = 0
if "retouch_key_counter" not in st.session_state: st.session_state.retouch_key_counter = 0

with st.sidebar:
    st.title("⚙️ Config")
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini Key Loaded")
    elif "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ Google Key Loaded")
    else:
        api_key = st.text_input("Gemini API Key", type="password")
    api_key = clean_key(api_key)

    if "JSONBIN_API_KEY" in st.secrets: st.caption("✅ Database Connected")
    else: st.warning("⚠️ Local Mode (DB Not Connected)")

st.title("💎 Jewelry AI Studio")
tab1, tab_retouch, tab2, tab3, tab4, tab5 = st.tabs(["✨ Gen Image", "🎨 Retouch", "🏷️ Bulk SEO", "📝 Writer", "📚 Library", "ℹ️ Models"])

# === TAB 1: GEN IMAGE (UPDATED) ===
with tab1:
    # State สำหรับเก็บรูปจาก Shopify ใน Tab นี้
    if "gen_shopify_imgs" not in st.session_state: st.session_state.gen_shopify_imgs = []

    c1, c2 = st.columns([1, 1.2])
    
    # --- COLUMN 1: INPUT ---
    with c1:
        st.subheader("1. Source Images")
        
        # A. Shopify Import
        with st.expander("🛍️ Import from Shopify", expanded=True):
            sh_secret_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            sh_secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            
            if sh_secret_shop and sh_secret_token:
                sh_gen_id = st.text_input("Product ID", key="gen_shopify_id")
                
                col_fetch, col_clear = st.columns([2, 1])
                if col_fetch.button("⬇️ Fetch Images", key="gen_fetch_btn"):
                    if not sh_gen_id: st.warning("Enter ID")
                    else:
                        with st.spinner("Downloading..."):
                            imgs, err = get_shopify_product_images(sh_secret_shop, sh_secret_token, sh_gen_id)
                            if imgs:
                                # Fetch Detail to get handle for URL
                                _, _, handle, _ = get_shopify_product_details(sh_secret_shop, sh_secret_token, sh_gen_id)
                                if handle:
                                    clean_shop = sh_secret_shop.replace("https://", "").replace("http://", "").strip()
                                    if not clean_shop.endswith(".myshopify.com"): clean_shop += ".myshopify.com"
                                    # Update 'post_url' session state directly
                                    st.session_state['post_url'] = f"https://{clean_shop}/products/{handle}"

                                st.session_state.gen_shopify_imgs = imgs
                                st.session_state['gen_upload_id'] = sh_gen_id # Sync Bottom ID
                                st.success(f"Loaded {len(imgs)} images")
                                st.rerun()
                            else: st.error(err)
                            
                if col_clear.button("❌ Clear", key="gen_clear_btn"):
                    st.session_state.gen_shopify_imgs = []
                    if 'post_url' in st.session_state: st.session_state['post_url'] = ""
                    st.rerun()
            else:
                st.info("Set Secrets to use Import")

        # B. Source Logic & Display
        images_to_send = []
        
        # Priority: Shopify > Manual
        if st.session_state.gen_shopify_imgs:
            images_to_send = st.session_state.gen_shopify_imgs
            st.info(f"Using {len(images_to_send)} images from Shopify")
            
            # --- DOWNLOAD ALL BUTTON (Specific for Gen Image) ---
            try:
                zip_gen = BytesIO()
                with zipfile.ZipFile(zip_gen, "w") as zf:
                    for i, img in enumerate(images_to_send):
                        buf = BytesIO()
                        img.save(buf, format="JPEG", quality=95)
                        zf.writestr(f"shopify_orig_{i+1}.jpg", buf.getvalue())
                
                st.download_button(
                    "💾 Download All Originals (.zip)",
                    data=zip_gen.getvalue(),
                    file_name="shopify_original_images.zip",
                    mime="application/zip"
                )
            except: pass
            # ----------------------------------------------------
            
        else:
            files = st.file_uploader("Upload Manual", accept_multiple_files=True, type=["jpg","png"], key="gen_up")
            images_to_send = [Image.open(f) for f in files] if files else []

        # Preview
        if images_to_send:
            cols = st.columns(4)
            for i, img in enumerate(images_to_send): cols[i%4].image(img, use_column_width=True)

    # --- COLUMN 2: SETTINGS (เหมือนเดิม) ---
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
                    # 1. Gen Image
                    with st.spinner("Generating Image & Tags..."):
                        d, e = generate_image(api_key, images_to_send, prompt_edit)
                        if d:
                            st.session_state.current_generated_image = d
                            st.session_state.image_generated_success = True
                            
                            # 2. Gen Tags (From Prompt)
                            tags_json, tags_err = generate_seo_tags_from_context(api_key, prompt_edit)
                            if tags_json:
                                parsed_tags = parse_json_response(tags_json)
                                if parsed_tags:
                                    st.session_state.gen_tags_result = parsed_tags
                                else:
                                    st.session_state.gen_tags_result = {}
                            else:
                                st.session_state.gen_tags_result = {}
                                
                            st.rerun()
                        else: st.error(e)

            if st.session_state.image_generated_success and st.session_state.current_generated_image:
                st.divider()
                st.subheader("✨ Result")
                # Full width image view
                st.image(st.session_state.current_generated_image, use_column_width=True)
                st.download_button("💾 Download Image", st.session_state.current_generated_image, "gen.jpg", "image/jpeg", type="secondary")

                st.divider()
                st.subheader("☁️ Upload to Shopify (Add New Image)")
                with st.container(border=True):
                    
                    # Display & Edit Tags
                    tags_data = st.session_state.get("gen_tags_result", {})
                    
                    default_filename = tags_data.get("file_name", "")
                    default_alt = tags_data.get("alt_tag", "")
                    
                    col_tags1, col_tags2 = st.columns(2)
                    final_filename = col_tags1.text_input("File Name", value=default_filename, help="SEO-optimized filename")
                    final_alt = col_tags2.text_input("Alt Tag", value=default_alt, help="Descriptive alt text")
                    
                    # Auto-load Secrets
                    s_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
                    s_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
                    
                    # พยายามดึง ID จากช่อง Import ด้านบนมาใส่ให้
                    default_id = st.session_state.get("gen_shopify_id", "")
                    
                    col_u1, col_u2 = st.columns([3, 1])
                    # Use session state key directly for value update
                    u_prod_id = col_u1.text_input("Product ID", value=default_id, key="gen_upload_id", label_visibility="collapsed")
                    
                    if col_u2.button("🚀 Upload", type="primary", use_container_width=True):
                        if not s_shop or not s_token:
                            st.error("Missing Shopify Secrets")
                        elif not u_prod_id:
                            st.warning("Enter Product ID")
                        else:
                            with st.spinner("Uploading to Shopify..."):
                                # เรียกใช้ฟังก์ชันใหม่สำหรับ ADD รูป (POST) พร้อม Tags
                                success, msg = add_single_image_to_shopify(
                                    s_shop, s_token, u_prod_id, 
                                    st.session_state.current_generated_image,
                                    file_name=final_filename,
                                    alt_tag=final_alt
                                )
                                if success: st.success(msg)
                                else: st.error(msg)

# === TAB 1.5: RETOUCH IMAGES (UPDATED WITH SHOPIFY IMPORT) ===
with tab_retouch:
    st.header("🎨 Retouch (via Gemini)")
    st.caption("Upload raw product photos OR Import directly from Shopify.")
    
    # State สำหรับเก็บรูปจาก Shopify (ป้องกันการหายเวลากดปุ่มอื่น)
    if "shopify_fetched_imgs" not in st.session_state:
        st.session_state.shopify_fetched_imgs = []

    rt_key_id = st.session_state.retouch_key_counter
    
    rt_c1, rt_c2 = st.columns([1, 1.2])
    
    # --- COLUMN 1: INPUT SOURCE ---
    with rt_c1:
        st.subheader("1. Input Images")
        
        # A. Shopify Import Section
        with st.expander("🛍️ Import from Shopify (Optional)", expanded=True):
            # Auto-load Secrets
            sh_secret_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            sh_secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            
            if sh_secret_shop and sh_secret_token:
                st.success("✅ Shopify Connected")
                # ช่องกรอก ID สำหรับดึงรูป (จะใช้ค่านี้ไปเป็น default ในช่อง upload ด้วย)
                sh_imp_id = st.text_input("Product ID to Fetch", key=f"imp_id_{rt_key_id}")
                
                c_fetch, c_clear = st.columns([2,1])
                if c_fetch.button("⬇️ Fetch Images"):
                    if not sh_imp_id:
                        st.warning("Please enter Product ID")
                    else:
                        with st.spinner("Downloading images from Shopify..."):
                            imgs, err = get_shopify_product_images(sh_secret_shop, sh_secret_token, sh_imp_id)
                            if imgs:
                                st.session_state.shopify_fetched_imgs = imgs
                                st.success(f"Loaded {len(imgs)} images!")
                                st.rerun()
                            else:
                                st.error(err)
                
                if c_clear.button("❌ Clear"):
                    st.session_state.shopify_fetched_imgs = []
                    st.rerun()
            else:
                st.info("Set SHOPIFY_SHOP_URL and SHOPIFY_ACCESS_TOKEN in secrets to use this feature.")

        # B. Determine Source (Shopify vs Upload)
        rt_imgs = []
        source_type = ""
        
        # Priority 1: Images from Shopify
        if st.session_state.shopify_fetched_imgs:
            rt_imgs = st.session_state.shopify_fetched_imgs
            source_type = "Shopify"
            st.info(f"📂 Using {len(rt_imgs)} images from Shopify Product")
            
            # Feature: Download Original Images (ที่ User ขอมา)
            try:
                zip_orig = BytesIO()
                with zipfile.ZipFile(zip_orig, "w") as zf:
                    for i, img in enumerate(rt_imgs):
                        # Save as JPEG
                        buf = BytesIO()
                        img.save(buf, format="JPEG", quality=95)
                        zf.writestr(f"original_shopify_{i+1}.jpg", buf.getvalue())
                
                st.download_button(
                    "💾 Download All Originals (.zip)",
                    data=zip_orig.getvalue(),
                    file_name="shopify_original_images.zip",
                    mime="application/zip"
                )
            except Exception as e: st.error(f"Zip Error: {e}")

        # Priority 2: Manual Upload (ถ้าไม่ได้ดึงจาก Shopify)
        else:
            rt_files = st.file_uploader("Upload Manual Images", accept_multiple_files=True, type=["jpg", "png"], key=f"rt_up_{rt_key_id}")
            if rt_files:
                rt_imgs = [Image.open(f) for f in rt_files]
                source_type = "Upload"
        
        # Preview Images
        if rt_imgs:
            with st.expander(f"📸 View Input ({len(rt_imgs)} images)", expanded=False):
                cols = st.columns(4)
                for i, img in enumerate(rt_imgs):
                    cols[i%4].image(img, use_column_width=True, caption=f"In #{i+1}")
        else:
            st.warning("Waiting for images...")

    # --- COLUMN 2: PROCESS & OUTPUT ---
    with rt_c2:
        st.subheader("2. Prompt Settings")
        lib = st.session_state.library
        rt_cats = list(set(p.get('category','Other') for p in lib)) if lib else []
        
        default_cat_index = rt_cats.index("Retouch") if "Retouch" in rt_cats else 0
        rt_sel_cat = st.selectbox("Category", rt_cats, index=default_cat_index, key=f"rt_cat_{rt_key_id}") if rt_cats else None
        
        rt_filtered = [p for p in lib if p.get('category') == rt_sel_cat]
        if rt_filtered:
            rt_style = st.selectbox("Style", rt_filtered, format_func=lambda x: x.get('name','Unknown'), key=f"rt_style_{rt_key_id}")
            
            style_tracker_key = f"last_rt_style_{rt_key_id}"
            if style_tracker_key not in st.session_state:
                st.session_state[style_tracker_key] = rt_style['id']
                
            style_changed = False
            if st.session_state[style_tracker_key] != rt_style['id']:
                style_changed = True
                st.session_state[style_tracker_key] = rt_style['id']
            
            rt_vars = [v.strip() for v in rt_style.get('variables','').split(",") if v.strip()]
            rt_user_vals = {v: st.text_input(v, key=f"rt_var_{v}_{rt_key_id}") for v in rt_vars}
            
            rt_final_prompt = rt_style.get('template','')
            for k, v in rt_user_vals.items(): rt_final_prompt = rt_final_prompt.replace(f"{{{k}}}", v)
            
            prompt_key = f"rt_prompt_{rt_key_id}"
            if style_changed: st.session_state[prompt_key] = rt_final_prompt
            
            st.write("✏️ **Retouch Instruction:**")
            rt_prompt_edit = st.text_area("Instruction", value=rt_final_prompt, height=100, key=prompt_key)
            
            c_rt1, c_rt2 = st.columns([1, 1])
            run_retouch = c_rt1.button("🚀 Run Batch Retouch", type="primary", disabled=(not rt_imgs))
            clear_retouch = c_rt2.button("🔄 Start Over", key="clear_retouch")
            
            if clear_retouch:
                st.session_state.retouch_results = None
                st.session_state.seo_name_result = None
                st.session_state.shopify_fetched_imgs = [] # Clear fetched images too
                st.session_state.retouch_key_counter += 1
                st.rerun()
            
            if run_retouch:
                if not api_key:
                    st.error("Missing Gemini API Key!")
                else:
                    rt_temp_results = []
                    rt_pbar = st.progress(0)
                    
                    for i, img in enumerate(rt_imgs):
                        with st.spinner(f"Processing Image #{i+1} with Gemini..."):
                            gen_img_bytes, err = generate_image(api_key, [img], rt_prompt_edit)
                            rt_pbar.progress((i+1)/len(rt_imgs))
                            
                            if gen_img_bytes:
                                rt_temp_results.append(gen_img_bytes)
                            else:
                                st.error(f"Failed Image #{i+1}: {err}")
                                rt_temp_results.append(None)
                                
                    st.session_state.retouch_results = rt_temp_results
                    st.success("Batch Processing Complete!")
                    st.rerun()

    if st.session_state.retouch_results:
        st.divider()
        st.subheader("🎨 Retouched Results (Gemini)")
        
        # Download All Retouched
        try:
            zip_buf = BytesIO()
            has_files = False
            with zipfile.ZipFile(zip_buf, "w") as zf:
                for i, res_bytes in enumerate(st.session_state.retouch_results):
                    if res_bytes:
                        zf.writestr(f"retouched_{i+1}.jpg", res_bytes)
                        has_files = True
            
            if has_files:
                st.download_button(
                    label="📦 Download All Retouched (.zip)",
                    data=zip_buf.getvalue(),
                    file_name="all_retouched_images.zip",
                    mime="application/zip",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error creating zip: {e}")

        cols = st.columns(3)
        for i, res_bytes in enumerate(st.session_state.retouch_results):
            with cols[i % 3]:
                st.write(f"**Result #{i+1}**")
                if res_bytes:
                    st.image(res_bytes, use_column_width=True)
                else: st.error("Failed")

        # --- AUTOMATION: UPLOAD TO SHOPIFY (REPLACE ALL) ---
        st.markdown("---")
        st.subheader("🚀 Automation: Upload to Shopify")
        st.caption("⚠️ การกดปุ่มนี้จะ **ลบรูปเดิมทั้งหมด** บน Shopify และแทนที่ด้วยชุดรูป Retouch นี้")
        
        with st.container(border=True):
            # Auto-load Secrets
            rt_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            rt_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            
            # พยายามดึง ID จากช่อง Import ด้านบน (auto-fill default value)
            # key "imp_id_{rt_key_id}" มาจาก loop ต้องดึงค่าให้ถูก
            current_imp_id = st.session_state.get(f"imp_id_{rt_key_id}", "")
            
            col_rt_u1, col_rt_u2 = st.columns([2, 1])
            rt_prod_id = col_rt_u1.text_input("Target Product ID", value=current_imp_id, key="rt_upload_id")
            
            if col_rt_u2.button("☁️ Upload All & Replace", type="primary", use_container_width=True):
                if not rt_shop or not rt_token:
                    st.error("Missing Secrets")
                elif not rt_prod_id:
                    st.warning("Enter Product ID")
                elif not any(st.session_state.retouch_results):
                    st.warning("No images to upload")
                else:
                    with st.spinner(f"Uploading {len(st.session_state.retouch_results)} images..."):
                        success, msg = upload_only_images_to_shopify(
                            rt_shop, rt_token, rt_prod_id, 
                            st.session_state.retouch_results
                        )
                        if success: st.success(msg); st.balloons()
                        else: st.error(msg)
    
    # ... (ส่วน SEO Name & Slug Generator เดิมของคุณ ให้คงไว้ต่อท้ายตรงนี้ได้เลย) ...
    # ========================================================
    # NEW FEATURE: SEO PRODUCT NAME & SLUG GENERATOR
    # ========================================================
    st.markdown("---")
    st.subheader("🛍️ SEO Product Name & Slug Generator")
    # ... (โค้ดส่วนนี้เหมือนเดิม ใช้ rt_imgs ต่อได้เลย เพราะเรา override มาแล้ว) ...
    # 1. Image Source Logic
    target_images_for_seo = []
    source_label = ""
    
    if st.session_state.retouch_results and any(st.session_state.retouch_results):
        target_images_for_seo = [x for x in st.session_state.retouch_results if x is not None]
        source_label = "✅ Using Retouched Images"
    elif rt_imgs:
        target_images_for_seo = rt_imgs
        source_label = f"✅ Using {source_type} Images"
    else:
        source_label = "⚠️ No images available"

    c_seo1, c_seo2 = st.columns([1, 1])
    with c_seo1:
        user_product_desc = st.text_input("Basic Product Description", placeholder="e.g., sterling silver bracelet", key=f"seo_desc_{rt_key_id}")
        st.write(f"Source: {source_label}")
        
        if st.button("✨ Analyze Name & Slug"):
            if not api_key: st.error("Missing API Key")
            elif not target_images_for_seo: st.warning("No images.")
            elif not user_product_desc: st.warning("Enter description.")
            else:
                with st.spinner("Analyzing SEO..."):
                    seo_json, seo_err = generate_seo_name_slug(api_key, target_images_for_seo, user_product_desc)
                    if seo_json:
                        res_dict = parse_json_response(seo_json)
                        if res_dict: st.session_state.seo_name_result = res_dict
                        else: st.error("Failed to parse"); st.code(seo_json)
                    else: st.error(seo_err)

    with c_seo2:
        if st.session_state.seo_name_result:
            res = st.session_state.seo_name_result
            st.success("Analysis Complete!")
            st.write("**Product Name:**")
            st.text_input("Name", value=res.get("product_name", ""), label_visibility="collapsed", key=f"res_name_{rt_key_id}")
            st.write("**URL Slug:**")
            st.code(res.get("url_slug", ""), language="text")


# === TAB 2: BULK SEO ===
with tab2:
    st.header("🏷️ Bulk SEO Tags")
    bulk_key_id = st.session_state.bulk_key_counter
    
    bc1, bc2 = st.columns([1, 1.5])
    with bc1:
        bfiles = st.file_uploader("Upload Images", accept_multiple_files=True, key=f"bulk_up_{bulk_key_id}")
        bimgs = [Image.open(f) for f in bfiles] if bfiles else []
        if bimgs:
            st.success(f"{len(bimgs)} images selected")
            with st.expander("📸 Preview", expanded=False):
                cols = st.columns(4)
                for i, img in enumerate(bimgs):
                    cols[i%4].image(img, use_column_width=True, caption=f"Img #{i+1}")

    with bc2:
        burl = st.text_input("Product URL:", key=f"bulk_url_{bulk_key_id}")
        c_btn1, c_btn2 = st.columns([1, 1])
        run_batch = c_btn1.button("🚀 Run Batch", type="primary", disabled=(not bimgs))
        clear_batch = c_btn2.button("🔄 Start Over", key="clear_bulk")

        if clear_batch:
            st.session_state.bulk_results = None
            st.session_state.bulk_key_counter += 1
            st.rerun()

        if run_batch:
            if not api_key or not burl: st.error("Missing Info")
            else:
                pbar = st.progress(0)
                temp_results = []
                for i, img in enumerate(bimgs):
                    with st.spinner(f"Processing Image #{i+1}..."):
                        txt, err = generate_seo_for_existing_image(api_key, img, burl)
                        pbar.progress((i+1)/len(bimgs))
                        if txt:
                            d = parse_json_response(txt)
                            if isinstance(d, list) and len(d) > 0: d = d[0]
                            if isinstance(d, dict):
                                temp_results.append(d)
                            else:
                                temp_results.append({"error": "Invalid format", "raw": txt})
                        else:
                            temp_results.append({"error": err})
                st.session_state.bulk_results = temp_results
                st.success("Done!")
                st.rerun()

    if st.session_state.bulk_results and bimgs:
        st.divider()
        for i, res in enumerate(st.session_state.bulk_results):
            if i < len(bimgs):
                with st.container():
                    rc1, rc2 = st.columns([1, 3])
                    with rc1:
                        st.image(bimgs[i], width=150, caption=f"Img #{i+1}")
                    with rc2:
                        if "error" in res:
                            st.error(f"Error: {res.get('error')}")
                            if "raw" in res: st.code(res['raw'])
                        else:
                            st.write("**File Name:**")
                            st.code(res.get('file_name', ''), language="text")
                            st.write("**Alt Tag:**")
                            st.code(res.get('alt_tag', ''), language="text")
                    st.divider()

# === TAB 3: WRITER (FIXED ERROR) ===
with tab3:
    st.header("📝 Product Writer")
    writer_key_id = st.session_state.writer_key_counter
    
    # Init Session State สำหรับ Writer Import
    if "writer_shopify_imgs" not in st.session_state: st.session_state.writer_shopify_imgs = []
    
    # Key สำหรับ Text Area เพื่อให้เรา Update ค่าได้
    text_area_key = f"w_raw_{writer_key_id}"
    
    c1, c2 = st.columns([1, 1.2])
    
    # --- COLUMN 1: INPUT ---
    with c1:
        # A. Shopify Import Section
        with st.expander("🛍️ Import from Shopify (Images & Desc)", expanded=True):
            sh_secret_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
            sh_secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
            
            if sh_secret_shop and sh_secret_token:
                sh_writer_id = st.text_input("Product ID", key="writer_shopify_id")
                
                col_w_fetch, col_w_clear = st.columns([2, 1])
                
                if col_w_fetch.button("⬇️ Fetch All", key="writer_fetch_btn"):
                    if not sh_writer_id: st.warning("Enter ID")
                    else:
                        with st.spinner("Fetching Data..."):
                            # 1. Fetch Images
                            imgs, err_img = get_shopify_product_images(sh_secret_shop, sh_secret_token, sh_writer_id)
                            
                            # 2. Fetch Description (FIXED LINE: รับค่าตัวแปรที่ 3 เป็น _ หรือ handle ก็ได้)
                            desc_html, title, _, err_desc = get_shopify_product_details(sh_secret_shop, sh_secret_token, sh_writer_id)
                            
                            if imgs:
                                st.session_state.writer_shopify_imgs = imgs
                            
                            if desc_html is not None: 
                                # --- ดึงเฉพาะ Description ไม่เอา Title ---
                                clean_desc = remove_html_tags(desc_html)
                                combined_text = clean_desc 
                                # ----------------------------------------
                                
                                st.session_state[text_area_key] = combined_text
                                
                            st.success("Loaded!")
                            st.rerun()
                            
                if col_w_clear.button("❌ Clear", key="writer_clear_btn"):
                    st.session_state.writer_shopify_imgs = []
                    if text_area_key in st.session_state:
                        st.session_state[text_area_key] = ""
                    st.rerun()
                    
        # B. Image Handling
        writer_imgs = []
        if st.session_state.writer_shopify_imgs:
            writer_imgs = st.session_state.writer_shopify_imgs
            st.info(f"Using {len(writer_imgs)} images from Shopify (No Download)")
        else:
            files = st.file_uploader("Images (Optional)", type=["jpg", "png"], accept_multiple_files=True, key=f"w_img_{writer_key_id}")
            writer_imgs = [Image.open(f) for f in files] if files else []
        
        if writer_imgs:
            with st.expander("📸 Image Preview", expanded=False):
                cols = st.columns(4)
                for i, img in enumerate(writer_imgs):
                    cols[i%4].image(img, use_column_width=True, caption=f"#{i+1}")

        # C. Text Input
        raw = st.text_area("Paste Details:", height=300, key=text_area_key)
        
        wb1, wb2 = st.columns([1, 1])
        run_write = wb1.button("🚀 Generate Content", type="primary")
        clear_write = wb2.button("🔄 Start Over", key="clear_writer")
        
        if clear_write:
            st.session_state.writer_result = None
            st.session_state.writer_shopify_imgs = []
            st.session_state.writer_key_counter += 1
            st.rerun()

    # --- COLUMN 2: OUTPUT & AUTOMATION ---
    with c2:
        if run_write:
            if not api_key or not raw: st.error("Missing Info")
            else:
                with st.spinner("Writing & Analyzing Images..."):
                    json_txt, err = generate_full_product_content(api_key, writer_imgs, raw)
                    if json_txt:
                        d = parse_json_response(json_txt)
                        if isinstance(d, list) and len(d) > 0: d = d[0]
                        if isinstance(d, dict):
                            st.session_state.writer_result = d
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

            # --- AUTOMATION SECTION ---
            st.markdown("---")
            st.subheader("🚀 Automation: Publish to Shopify")
            
            with st.container(border=True):
                st.info("ℹ️ ระบบจะอัปเดต: Title, Description (HTML), Meta Title/Desc และรูปภาพ (ถ้าเลือก)")
                
                secret_shop = st.secrets.get("SHOPIFY_SHOP_URL")
                secret_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN")
                
                s_shop = None
                s_token = None
                s_prod_id = None
                
                if secret_shop and secret_token:
                    col_info, col_input = st.columns([1, 1])
                    with col_info:
                        st.success("✅ Credentials Loaded from Secrets")
                        s_shop = secret_shop
                        s_token = secret_token
                    with col_input:
                        default_id = st.session_state.get("writer_shopify_id", "")
                        s_prod_id = st.text_input("Product ID", value=default_id, help="ID สินค้า")
                else:
                    st.warning("⚠️ Credentials Required")
                    c_x1, c_x2, c_x3 = st.columns(3)
                    s_shop = c_x1.text_input("Shop URL")
                    s_token = c_x2.text_input("Token", type="password")
                    s_prod_id = c_x3.text_input("Product ID")

                st.write("**Options:**")
                # Default เป็น Checked (True)
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

