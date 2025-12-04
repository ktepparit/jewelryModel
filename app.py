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

1. **Opening Paragraph:** บอก Google และผู้ใช้ให้ชัดเจนว่าหน้านี้คืออะไร (เน้น Primary Keyword + Semantic 1-2 คำ)
2. **Body Content:** เล่าเรื่องราว, ดีไซน์, สัญลักษณ์ (กระจาย Semantic Keywords อย่างเป็นธรรมชาติ)
3. **Specifications:** ใช้ Bullet Points (<ul><li>) เน้นวัสดุ และ **ต้องระบุ Dimension (ขนาด) และ Weight (น้ำหนัก)** ถ้ามีข้อมูล
4. **FAQ Section:** ตอบข้อสงสัย (ใช้ Long-tail keywords)

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

# Default Data (ใช้กรณีต่อ DB ไม่ติด)
DEFAULT_PROMPTS = [
    {
        "id": "p1", "name": "Luxury Hand (Ring)", "category": "Ring",
        "template": "A realistic close-up of a female hand model wearing a ring with {face_size} face size, soft studio lighting, elegant jewelry photography.",
        "variables": "face_size",
        "sample_url": "[https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Ring_render.jpg/320px-Ring_render.jpg](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Ring_render.jpg/320px-Ring_render.jpg)"
    },
    {
        "id": "p2", "name": "Streetwear Necklace", "category": "Necklace",
        "template": "A fashion portrait of a model wearing a {length} necklace, streetwear outfit, urban background, high detailed texture.",
        "variables": "length",
        "sample_url": "[https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Necklace_1.jpg/320px-Necklace_1.jpg](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Necklace_1.jpg/320px-Necklace_1.jpg)"
    }
]

# --- 2. CLOUD DATABASE FUNCTIONS (Improved Debugging) ---
def get_prompts():
    # 1. เช็คว่ามี Secrets ไหม
    if "JSONBIN_API_KEY" not in st.secrets or "JSONBIN_BIN_ID" not in st.secrets:
        st.session_state.db_status = "❌ No Secrets Found"
        return DEFAULT_PROMPTS

    try:
        API_KEY = st.secrets["JSONBIN_API_KEY"]
        BIN_ID = st.secrets["JSONBIN_BIN_ID"]
        
        # ใช้ /latest เพื่อเอาข้อมูลล่าสุด
        url = f"[https://api.jsonbin.io/v3/b/](https://api.jsonbin.io/v3/b/){BIN_ID}/latest"
        headers = {
            "X-Master-Key": API_KEY,
            "X-Bin-Meta": "false" # ขอข้อมูลเพียวๆ ไม่เอา Metadata
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.db_status = "✅ Connected"
            
            # 2. เช็คโครงสร้างข้อมูล (List หรือ Dict)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "record" in data:
                return data["record"]
            else:
                st.session_state.db_status = f"⚠️ Unknown JSON Format: {type(data)}"
                return DEFAULT_PROMPTS
        else:
            # 3. แจ้ง Error Code ชัดเจน
            st.session_state.db_status = f"❌ API Error: {response.status_code} - {response.reason}"
            return DEFAULT_PROMPTS
            
    except Exception as e:
        st.session_state.db_status = f"❌ Exception: {str(e)}"
        return DEFAULT_PROMPTS

def save_prompts(data):
    if "JSONBIN_API_KEY" in st.secrets and "JSONBIN_BIN_ID" in st.secrets:
        try:
            API_KEY = st.secrets["JSONBIN_API_KEY"]
            BIN_ID = st.secrets["JSONBIN_BIN_ID"]
            url = f"[https://api.jsonbin.io/v3/b/](https://api.jsonbin.io/v3/b/){BIN_ID}"
            headers = {"Content-Type": "application/json", "X-Master-Key": API_KEY}
            
            # ส่ง List ตรงๆ
            response = requests.put(url, json=data, headers=headers)
            if response.status_code != 200:
                st.error(f"Save failed: {response.text}")
        except Exception as e:
            st.error(f"Save exception: {e}")
    else:
        st.warning("Cannot save: No Database credentials found.")

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
    except:
        return None

def safe_st_image(url, width=None):
    """ฟังก์ชันแสดงรูปภาพแบบปลอดภัย ไม่ให้แอพพังถ้ารูปเสีย"""
    if not url: return
    try:
        # ลองตรวจสอบว่าเป็น URL หรือไม่
        if url.startswith("http"):
            st.image(url, width=width)
        else:
            st.caption("Invalid Image URL")
    except Exception as e:
        # ถ้าโหลดไม่ได้จริงๆ ให้แสดง Text แทน
        st.warning(f"⚠️ Cannot load preview image")

# Function 1: Gen Image
def generate_image(api_key, image_list, prompt):
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_IMAGE_GEN}:generateContent?key={api_key}"
    parts = [{"text": f"Instruction: {prompt} \nConstraint: Keep the jewelry products in the input images EXACTLY as they are. Analyze all images to
