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

        คีย์เวิร์ดหลัก (Primary Keyword): เน้นที่คีย์เวิร์ดหลัก ให้ชัดเจน

        Semantic Keyword ที่สำคัญที่สุด 1-2 คำ: 

2. ส่วนกลางของเนื้อหา (Body of the Content)

    เป้าหมาย: เล่าเรื่องราว, อธิบายดีไซน์, บอกคุณสมบัติ

    สิ่งที่ควรวาง: นี่คือพื้นที่ที่ดีที่สุดในการกระจาย Semantic Keywords ส่วนใหญ่

    ส่วนที่เล่าถึงที่มา/แรงบันดาลใจ: 

    ส่วนที่อธิบายสัญลักษณ์: 

3. ส่วนคุณสมบัติ (Specifications / Beautiful Bullet Points)

    เป้าหมาย: ให้ข้อมูลทางเทคนิคที่ชัดเจน

    สิ่งที่ควรวาง: เหมาะสำหรับคำที่เกี่ยวกับวัสดุ เช่น 925 sterling silver, solid silver, handcrafted, oxidized finish

4. ส่วนคำถามที่พบบ่อย (FAQ Section)

    เป้าหมาย: ตอบข้อสงสัยและให้ข้อมูลเพิ่มเติม

    สิ่งที่ควรวาง: เป็นโอกาสที่ดีในการใช้คีย์เวิร์ดแบบยาวๆ (Long-tail keywords) และ Semantic Keywords ที่เกี่ยวข้อง

** คีย์เวิร์ดรอง : กระจายคีย์เวิร์ดรอง ไปในเนื้อหาส่วนอื่นๆ อย่างเป็นธรรมชาติ 

** คีย์เวิร์ดหมวดหมู่): ใช้คำนี้เพื่อสร้างการเชื่อมโยงและอ้างอิงถึงหมวดหมู่ที่ใหญ่ขึ้น และเป็นโอกาสที่ดีในการสร้าง Internal Link กลับไปยังหน้า Collection

โดยลักษณะของการเขียนจัดเรียงประมาณนี้ครับ

Product Overview

Key Features at a Glance or Key Features & Benefits or  What Makes This Special ให้คุณเลือกใช้ได้ตามเหมาะสม

Frequently Asked Questions (FAQ) 

รวมทั้งช่วยเขียน Google SEO-optimized meta title (approximately 60 characters)  and Google SEO-optimized meta description รวมทั้งช่วยเขียน SEO-optimized image file name พร้อมกับ image alt tag สำหรับ product นี้สำหรับทุก images และ url slug


ตัวอย่างการเพิ่มคำที่เกี่ยวข้องโดยให้เพิ่ม Semantic Keywords เข้าไปเช่น ถ้าสมมติกำหนดให้คีย์เวิร์ดหลักคือ "medusa ring" เพิ่ม Semantic Keywords คือคำที่เกี่ยวข้องกับตำนานเมดูซ่าเข้าไปในเนื้อหาเพื่อสร้างความเชี่ยวชาญในหัวข้อ (Topical Authority) เช่น Greek mythology, serpent hair, petrifying gaze, goddess Athena เป็นต้น 

โดยกระจาย Semantic Keywords ไปทั่วทั้งบทความอย่างเป็นธรรมชาติ

หลักการทำงานของ Semantic Keywords

เป้าหมายของการใช้ Semantic Keywords คือการสร้างเนื้อหาที่ "สมบูรณ์" และ "เป็นธรรมชาติ" ในหัวข้อนั้นๆ เพื่อแสดงให้ Google เห็นว่าคุณคือผู้เชี่ยวชาญจริง ไม่ใช่แค่รู้จักคีย์เวิร์ดหลักเพียงคำเดียว

การอัดแน่นในย่อหน้าเดียว: จะทำให้เนื้อหาอ่านดูไม่เป็นธรรมชาติ และอาจถูกมองว่าเป็นการพยายามยัดเยียดคีย์เวิร์ด (Keyword Stuffing) ซึ่งส่งผลเสียต่อ SEO ได้

การกระจายอย่างเป็นธรรมชาติ: จะทำให้บทความของคุณมีมิติ น่าอ่าน และที่สำคัญคือเป็นการส่งสัญญาณให้ Google เห็นว่าเนื้อหาของคุณมีคุณภาพและครอบคลุมหัวข้อนั้นๆ อย่างลึกซึ้ง

หลักการทำงานของ Semantic Keywords

กลยุทธ์การวาง Semantic Keywords ที่ดีที่สุด (Best Placement Strategy)

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

# --- SHOPIFY HELPER FUNCTION ---
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
        
def clean_filename(name):
    if not name: return "N/A"
    clean = re.sub(r'[^a-zA-Z0-9\-\_\.]', '', str(name))
    return clean.rsplit('.', 1)[0]

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

# === TAB 1.5: RETOUCH IMAGES (GEMINI VERSION + DOWNLOAD ALL + SEO NAME) ===
with tab_retouch:
    st.header("🎨 Retouch (via Gemini)")
    st.caption("Upload raw product photos. Gemini will regenerate them based on your prompt (one by one).")
    
    rt_key_id = st.session_state.retouch_key_counter
    
    rt_c1, rt_c2 = st.columns([1, 1.2])
    with rt_c1:
        st.subheader("1. Input Images")
        rt_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png"], key=f"rt_up_{rt_key_id}")
        rt_imgs = [Image.open(f) for f in rt_files] if rt_files else []
        
        if rt_imgs:
            st.success(f"{len(rt_imgs)} images loaded.")
            with st.expander("View Input", expanded=False):
                cols = st.columns(4)
                for i, img in enumerate(rt_imgs):
                    cols[i%4].image(img, use_column_width=True, caption=f"Input #{i+1}")

    with rt_c2:
        st.subheader("2. Prompt Settings")
        lib = st.session_state.library
        rt_cats = list(set(p.get('category','Other') for p in lib)) if lib else []
        
        # --- MODIFIED LOGIC: Default to 'Retouch' category if exists ---
        default_cat_index = 0
        if "Retouch" in rt_cats:
            default_cat_index = rt_cats.index("Retouch")
        # -------------------------------------------------------------

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
                st.session_state.retouch_key_counter += 1
                st.rerun()
            
            if run_retouch:
                if not api_key:
                    st.error("Missing Gemini API Key!")
                else:
                    rt_temp_results = []
                    rt_pbar = st.progress(0)
                    
                    # --- LOOP: Process One Image at a Time ---
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
        
        # --- DOWNLOAD ALL BUTTON ---
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
                    label="📦 Download All Images (.zip)",
                    data=zip_buf.getvalue(),
                    file_name="all_retouched_images.zip",
                    mime="application/zip",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error creating zip: {e}")
        # ---------------------------

        cols = st.columns(3)
        for i, res_bytes in enumerate(st.session_state.retouch_results):
            with cols[i % 3]:
                st.write(f"**Result #{i+1}**")
                if res_bytes:
                    st.image(res_bytes, use_column_width=True)
                    st.download_button("Download", res_bytes, file_name=f"retouched_{i+1}.jpg", mime="image/jpeg", key=f"dl_rt_{i}")
                else: st.error("Failed")

    # ========================================================
    # NEW FEATURE: SEO PRODUCT NAME & SLUG GENERATOR
    # ========================================================
    st.markdown("---")
    st.subheader("🛍️ SEO Product Name & Slug Generator")
    st.caption("Auto-generate a catchy product name and clean URL slug based on the images.")

    # 1. Image Source Logic
    target_images_for_seo = []
    source_label = ""
    
    if st.session_state.retouch_results and any(st.session_state.retouch_results):
        # Use Retouched images (Bytes)
        target_images_for_seo = [x for x in st.session_state.retouch_results if x is not None]
        source_label = "✅ Using Retouched Images"
    elif rt_imgs:
        # Use Input images (PIL)
        target_images_for_seo = rt_imgs
        source_label = "✅ Using Input Images (Original)"
    else:
        source_label = "⚠️ No images available"

    # 2. Input & Button
    c_seo1, c_seo2 = st.columns([1, 1])
    with c_seo1:
        user_product_desc = st.text_input(
            "Basic Product Description",
            placeholder="e.g., sterling silver bracelet, gemstone ring",
            help="Enter a short keyword to help the AI."
        )
        st.write(f"Source: {source_label}")
        
        if st.button("✨ Analyze Name & Slug"):
            if not api_key: st.error("Missing API Key")
            elif not target_images_for_seo: st.warning("Please upload images first.")
            elif not user_product_desc: st.warning("Please enter a description.")
            else:
                with st.spinner("Analyzing SEO..."):
                    # Call new function
                    seo_json, seo_err = generate_seo_name_slug(api_key, target_images_for_seo, user_product_desc)
                    if seo_json:
                        res_dict = parse_json_response(seo_json)
                        if res_dict:
                            st.session_state.seo_name_result = res_dict
                        else:
                            st.error("Failed to parse result")
                            st.code(seo_json)
                    else:
                        st.error(seo_err)

    # 3. Results Display
    with c_seo2:
        if st.session_state.seo_name_result:
            res = st.session_state.seo_name_result
            st.success("Analysis Complete!")
            
            st.write("**Product Name:**")
            st.text_input("Name", value=res.get("product_name", ""), label_visibility="collapsed")
            
            st.write("**URL Slug:**")
            st.code(res.get("url_slug", ""), language="text")
    # ========================================================


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

# === TAB 3: WRITER ===
with tab3:
    st.header("📝 Product Writer")
    writer_key_id = st.session_state.writer_key_counter
    
    c1, c2 = st.columns([1, 1.2])
    with c1:
        files = st.file_uploader("Images (Optional)", type=["jpg", "png"], accept_multiple_files=True, key=f"w_img_{writer_key_id}")
        writer_imgs = [Image.open(f) for f in files] if files else []
        
        if writer_imgs:
            with st.expander("📸 Image Preview", expanded=False):
                cols = st.columns(4)
                for i, img in enumerate(writer_imgs):
                    cols[i%4].image(img, use_column_width=True, caption=f"#{i+1}")

        raw = st.text_area("Paste Details:", height=300, key=f"w_raw_{writer_key_id}")
        
        wb1, wb2 = st.columns([1, 1])
        run_write = wb1.button("🚀 Generate Content", type="primary")
        clear_write = wb2.button("🔄 Start Over", key="clear_writer")
        
        if clear_write:
            st.session_state.writer_result = None
            st.session_state.writer_key_counter += 1
            st.rerun()

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
                                
                                st.write("**File Name:**")
                                st.code(fname, language="text")
                                st.write("**Alt Tag:**")
                                st.code(atag, language="text")
                            else:
                                st.warning(f"⚠️ AI did not generate tags for Image #{i+1}")
                        st.divider()
            else:
                st.info("No images uploaded.")

            
        # ... (ต่อจากโค้ดเดิมใน Tab 3 ส่วน Loop แสดง Image SEO) ...

            st.markdown("---")
            st.subheader("🚀 Automation: Publish to Shopify")
            
            with st.container(border=True):
                st.info("ℹ️ ระบบจะอัปเดต: Title, Description (HTML), Meta Title/Desc และรูปภาพ (ถ้าเลือก)")
                
                # 1. Config Inputs (ดึงจาก Secrets ก่อน ถ้าไม่มีให้กรอกเอง)
                col_s1, col_s2, col_s3 = st.columns(3)
                
                default_shop = st.secrets.get("SHOPIFY_SHOP_URL", "")
                default_token = st.secrets.get("SHOPIFY_ACCESS_TOKEN", "")
                
                # Input: Shop URL
                s_shop = col_s1.text_input("Shop URL (.myshopify.com)", value=default_shop, help="เช่น your-shop.myshopify.com")
                
                # Input: Token
                s_token = col_s2.text_input("Access Token (shpat_...)", value=default_token, type="password")
                
                # Input: Product ID (ตาม Requirement ข้อ 1)
                s_prod_id = col_s3.text_input("Product ID", help="เลข ID สินค้าจาก URL หลังบ้าน Shopify เช่น 8472xxxx")
                
                # 2. Options (ตาม Requirement ข้อ 5)
                st.write("**Options:**")
                
                # Checkbox เลือกรูปภาพ (ตาม Requirement ข้อ 4 & 5)
                # ถ้าไม่ติ๊ก -> รูปใน Shopify จะเหมือนเดิม ไม่ถูกยุ่งเกี่ยว
                # ถ้าติ๊ก -> รูปเก่าจะหายไป แทนที่ด้วยรูปใหม่ + ชื่อไฟล์/Alt tag จาก AI
                enable_img_upload = st.checkbox("📷 Upload Images & Replace Existing", value=False, help="ถ้าเลือก: รูปเดิมบน Shopify จะถูกลบและแทนที่ด้วยรูปชุดนี้ พร้อมตั้งชื่อไฟล์และ Alt Tag ตามผลลัพธ์ AI")
                
                if enable_img_upload and not writer_imgs:
                    st.warning("⚠️ คุณเลือกอัปโหลดรูปภาพ แต่ยังไม่มีรูปในระบบ (Writer Images)")

                # 3. Submit Button
                if st.button("☁️ Update Product to Shopify Now", type="primary", use_container_width=True):
                    # Validation เบื้องต้น
                    if not s_shop or not s_token or not s_prod_id:
                        st.error("กรุณากรอกข้อมูล Shop URL, Token และ Product ID ให้ครบถ้วน")
                    elif not st.session_state.writer_result:
                        st.error("กรุณา Generate Content ก่อนกดส่ง")
                    else:
                        # เรียกใช้ฟังก์ชัน
                        with st.spinner("Connecting to Shopify... (Sending Data & Images)"):
                            success, msg = update_shopify_product_v2(
                                shop_url=s_shop,
                                access_token=s_token,
                                product_id=s_prod_id,
                                data=st.session_state.writer_result, # ข้อมูล Text/SEO (Requirement ข้อ 2)
                                images_pil=writer_imgs,     # รูปภาพ (Requirement ข้อ 4)
                                upload_images=enable_img_upload # Toggle (Requirement ข้อ 5)
                            )
                            
                            if success:
                                st.success(msg)
                                st.balloons()
                            else:
                                st.error(msg)

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


