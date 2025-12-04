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

# Model IDs Configuration
MODEL_IMAGE_GEN = "models/gemini-3-pro-image-preview"
MODEL_TEXT_SEO = "models/gemini-3-pro-preview"

# --- PROMPTS ---

# Prompt A: สำหรับ Gen SEO Post-Gen (Tab 1)
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

# Prompt B: สำหรับ Bulk SEO (Tab 2)
SEO_PROMPT_BULK_EXISTING = """
คุณคือ SEO specialist ที่มีประสบการณ์ 15-20 ปี ช่วยเขียน SEO-optimized image file name with image alt tags เป็นภาษาอังกฤษ สำหรับสินค้าของฉันตามแต่ละรูปที่แนบมาให้ โดยมีรายละเอียดของสินค้าตาม url นี้ {product_url} เพื่อให้ได้ติดอันดับที่ดีบน organic search engine โดยกลุ่มลูกค้าเป็นผู้สนใจสินค้าชนิดนี้

IMPORTANT: You MUST return the result in raw JSON format ONLY (no markdown backticks).
Structure:
{
  "file_name": "your-optimized-filename.jpg",
  "alt_tag": "Your optimized descriptive alt tag"
}
"""

# Prompt C: สำหรับ Product Writer (Tab 3 - NEW!)
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
  "url_slug": "recommended-url-slug",
  "meta_title": "SEO Meta Title (Max 60 chars)",
  "meta_description": "SEO Meta Description (Max 160 chars)",
  "product_title_h1": "Product Title (H1)",
  "html_content": "<p>Full HTML content...</p>",
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

# Default Data
DEFAULT_PROMPTS = [
    {
        "id": "p1", "name": "Luxury Hand (Ring)", "category": "Ring",
        "template": "A realistic close-up of a female hand model wearing a ring with {face_size} face size, soft studio lighting, elegant jewelry photography.",
        "variables": "face_size",
        "sample_url": "[https://images.unsplash.com/photo-1599643478518-17488fbbcd75?q=80&w=300&auto=format&fit=crop](https://images.unsplash.com/photo-1599643478518-17488fbbcd75?q=80&w=300&auto=format&fit=crop)"
    },
    {
        "id": "p2", "name": "Streetwear Necklace", "category": "Necklace",
        "template": "A fashion portrait of a model wearing a {length} necklace, streetwear outfit, urban background, high detailed texture.",
        "variables": "length",
        "sample_url": "[https://images.unsplash.com/photo-1611591437281-460bfbe1220a?q=80&w=300&auto=format&fit=crop](https://images.unsplash.com/photo-1611591437281-460bfbe1220a?q=80&w=300&auto=format&fit=crop)"
    }
]

# --- 2. CLOUD DATABASE FUNCTIONS (JsonBin.io) ---
def get_prompts():
    try:
        API_KEY = st.secrets["JSONBIN_API_KEY"]
        BIN_ID = st.secrets["JSONBIN_BIN_ID"]
        url = f"[https://api.jsonbin.io/v3/b/](https://api.jsonbin.io/v3/b/){BIN_ID}/latest"
        headers = {"X-Master-Key": API_KEY}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("record", DEFAULT_PROMPTS)
        return DEFAULT_PROMPTS
    except:
        return DEFAULT_PROMPTS

def save_prompts(data):
    try:
        API_KEY = st.secrets["JSONBIN_API_KEY"]
        BIN_ID = st.secrets["JSONBIN_BIN_ID"]
        url = f"[https://api.jsonbin.io/v3/b/](https://api.jsonbin.io/v3/b/){BIN_ID}"
        headers = {"Content-Type": "application/json", "X-Master-Key": API_KEY}
        requests.put(url, json=data, headers=headers)
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
    except:
        return None

# Function 1: Gen Image
def generate_image(api_key, image_list, prompt):
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_IMAGE_GEN}:generateContent?key={api_key}"
    parts = [{"text": f"Instruction: {prompt} \nConstraint: Keep the jewelry products in the input images EXACTLY as they are. Analyze all images to understand the 3D structure. Generate a realistic model wearing it."}]
    for img in image_list:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    try:
        res = requests.post(url, json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.3}}, headers={"Content-Type": "application/json"})
        if res.status_code != 200: return None, f"API Error: {res.text}"
        content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
        if "inline_data" in content: return base64.b64decode(content["inline_data"]["data"]), None
        if "inlineData" in content: return base64.b64decode(content["inlineData"]["data"]), None
        if "text" in content: return None, f"Model returned text instead of image: {content['text']}"
        return None, "Unknown response format."
    except Exception as e: return None, str(e)

# Function 2: Gen SEO Post-Gen
def generate_seo_tags_post_gen(api_key, product_url):
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_TEXT_SEO}:generateContent?key={api_key}"
    final_seo_prompt = SEO_PROMPT_POST_GEN.replace("{product_url}", product_url)
    payload = {
        "contents": [{"parts": [{"text": final_seo_prompt}]}],
        "generationConfig": {"temperature": 0.5, "responseMimeType": "application/json"}
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                if "text" in content: return content["text"], None
                return None, "No text returned from model."
            elif res.status_code == 503 and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
                continue
            else:
                return None, f"API Error {res.status_code}: {res.text}"
        except Exception as e:
             if attempt < max_retries - 1: time.sleep(2); continue
             return None, str(e)
    return None, "Unknown error after retries."

# Function 3: Bulk SEO
def generate_seo_for_existing_image(api_key, img_pil, product_url):
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_TEXT_SEO}:generateContent?key={api_key}"
    final_prompt = SEO_PROMPT_BULK_EXISTING.replace("{product_url}", product_url)
    
    payload = {
        "contents": [{"parts": [
            {"text": final_prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}}
        ]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 2048, "responseMimeType": "application/json"}
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                if "text" in content: return content["text"], None
                return None, "Model returned no text."
            elif res.status_code == 503 and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
                continue
            else:
                return None, f"API Error {res.status_code}: {res.text}"
        except Exception as e:
            if attempt < max_retries - 1: time.sleep(2); continue
            return None, str(e)
    return None, "Failed after retries."

# Function 4: Product Writer (NEW!)
def generate_product_content(api_key, img_pil, raw_input):
    url = f"[https://generativelanguage.googleapis.com/v1beta/](https://generativelanguage.googleapis.com/v1beta/){MODEL_TEXT_SEO}:generateContent?key={api_key}"
    final_prompt = SEO_PRODUCT_WRITER_PROMPT.replace("{raw_input}", raw_input)
    
    parts = [{"text": final_prompt}]
    if img_pil:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}})
    
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"}
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
                if "text" in content: return content["text"], None
                return None, "Model returned no text."
            elif res.status_code == 503 and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
                continue
            else:
                return None, f"API Error {res.status_code}: {res.text}"
        except Exception as e:
            if attempt < max_retries - 1: time.sleep(2); continue
            return None, str(e)
    return None, "Failed after retries."

def list_available_models(api_key):
    url = f"[https://generativelanguage.googleapis.com/v1beta/models?key=](https://generativelanguage.googleapis.com/v1beta/models?key=){api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("models", [])
        else:
            return None
    except:
        return None

# --- 4. UI LOGIC ---
if "library" not in st.session_state:
    st.session_state.library = get_prompts()
if "edit_target" not in st.session_state:
    st.session_state.edit_target = None

# --- SESSION STATE ---
if "image_generated_success" not in st.session_state:
    st.session_state.image_generated_success = False
if "current_generated_image" not in st.session_state:
    st.session_state.current_generated_image = None

with st.sidebar:
    st.title("⚙️ Config")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("API Key Ready")
    except:
        api_key = st.text_input("Gemini API Key", type="password")
    
    if "JSONBIN_API_KEY" in st.secrets:
        st.caption("✅ Database Connected")
    else:
        st.warning("⚠️ Local Mode (Data will reset)")

st.title("💎 Jewelry AI Studio")

# Create Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "✨ Generate Image", 
    "🏷️ Bulk SEO Tags", 
    "📝 AI Product Writer", 
    "📚 Library Manager", 
    "ℹ️ About Models"
])

# === TAB 1: GENERATE IMAGE ===
with tab1:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("1. Upload Reference")
        files = st.file_uploader("Upload Images for Gen", accept_multiple_files=True, type=["jpg", "png", "jpeg"], key="gen_upload")
        images_to_send = [Image.open(f) for f in files] if files else []
        if images_to_send:
            st.caption(f"Selected {len(images_to_send)} images:")
            cols = st.columns(4)
            for i, img in enumerate(images_to_send): cols[i%4].image(img, use_column_width=True)

    with c2:
        st.subheader("2. Settings")
        lib = st.session_state.library
        cats = list(set(p['category'] for p in lib)) if lib else []
        sel_cat = st.selectbox("Category", cats) if cats else None
        
        filtered = [p for p in lib if p['category'] == sel_cat]
        if filtered:
            sel_style = st.selectbox("Style", filtered, format_func=lambda x: x['name'])
            # Safe Image Loading
            try:
                if sel_style.get("sample_url"): st.image(sel_style["sample_url"], width=100)
            except: st.caption("No Preview")
            
            vars_list = [v.strip() for v in sel_style['variables'].split(",") if v.strip()]
            user_vals = {v: st.text_input(v, placeholder="e.g. Gold") for v in vars_list}
            
            final_prompt = sel_style['template']
            for k, v in user_vals.items(): final_prompt = final_prompt.replace(f"{{{k}}}", v)
            
            st.write("✏️ **Edit Prompt:**")
            prompt_edit = st.text_area("Instruction", value=final_prompt, height=100)
            
            if st.button("🚀 GENERATE IMAGE", type="primary", use_container_width=True):
                if not api_key or not images_to_send:
                    st.error("Check Key & Images")
                    st.session_state.image_generated_success = False
                    st.session_state.current_generated_image = None
                else:
                    with st.spinner(f"Generating Image ({MODEL_IMAGE_GEN})..."):
                        d, e = generate_image(api_key, images_to_send, prompt_edit)
                        if d:
                            st.session_state.current_generated_image = d
                            st.session_state.image_generated_success = True
                            st.rerun()
                        else:
                            st.error(e)
                            st.session_state.image_generated_success = False

            if st.session_state.image_generated_success and st.session_state.current_generated_image:
                st.divider()
                st.subheader("🎉 Result")
                st.image(st.session_state.current_generated_image, use_column_width=True)
                st.download_button("Download Image", st.session_state.current_generated_image, "gen.jpg", "image/jpeg", type="primary")
                
                st.divider()
                st.subheader("🌍 SEO Tools (Post-Generation)")
                product_url_input = st.text_input("Paste Product URL here:", placeholder="[https://yourshop.com/product/](https://yourshop.com/product/)...", key="post_gen_url")
                
                if st.button("✨ Gen Tags for New Image"):
                    if not product_url_input:
                        st.warning("Please enter a Product URL first.")
                    else:
                        with st.spinner(f"Consulting SEO Specialist AI ({MODEL_TEXT_SEO})..."):
                            seo_text_json, seo_err = generate_seo_tags_post_gen(api_key, product_url_input)
                            
                            if seo_text_json:
                                seo_data = parse_json_response(seo_text_json)
                                if seo_data:
                                    with st.expander("✅ SEO Results", expanded=True):
                                        st.write("📄 **Image File Name:**")
                                        st.code(seo_data.get('file_name', 'N/A'), language="text")
                                        
                                        st.write("🏷️ **Alt Tag:**")
                                        st.code(seo_data.get('alt_tag', 'N/A'), language="text")
                                else:
                                    st.error("Failed to parse JSON. Raw output:")
                                    st.code(seo_text_json)
                            else:
                                st.error(f"SEO Generation Failed: {seo_err}")

        else: st.warning("Library empty.")

# === TAB 2: BULK SEO TAGS ===
with tab2:
    st.header("🏷️ Generate SEO Tags for Existing Images")
    
    bc1, bc2 = st.columns([1, 1.5])
    
    with bc1:
        st.subheader("1. Upload Existing Images")
        bulk_files = st.file_uploader("Choose images (Max 10 recommended)", accept_multiple_files=True, type=["jpg", "png", "jpeg"], key="bulk_seo_upload")
        bulk_images = [Image.open(f) for f in bulk_files] if bulk_files else []
        
        if bulk_images:
            st.success(f"✅ Ready! {len(bulk_images)} images selected.")
            st.caption("Preview:")
            cols_preview = st.columns(4)
            for i, img_pil in enumerate(bulk_images):
                 cols_preview[i % 4].image(img_pil, use_column_width=True, caption=f"#{i+1}")
        
    with bc2:
        st.subheader("2. Product Details & Run")
        bulk_url = st.text_input("Product URL (ใช้เป็นข้อมูลอ้างอิงสำหรับทุกรูป):", placeholder="[https://yourshop.com/product/](https://yourshop.com/product/)...", key="bulk_seo_url")
        
        run_bulk_btn = st.button("🚀 Run SEO Specialist AI (Batch)", type="primary", use_container_width=True, disabled=(not bulk_images))
        
    st.divider()
    st.subheader("📝 Results")

    if run_bulk_btn:
        if not api_key:
            st.error("Please provide API Key in settings first.")
        elif not bulk_url:
            st.error("Please provide a Product URL.")
        elif not bulk_images:
             st.warning("Please upload images first.")
        else:
            progress_bar = st.progress(0)
            results_container = st.container()

            for i, img_pil in enumerate(bulk_images):
                with st.spinner(f"Analyzing Image {i+1}/{len(bulk_images)} with {MODEL_TEXT_SEO}..."):
                    seo_text_json, error = generate_seo_for_existing_image(api_key, img_pil, bulk_url)
                    progress_bar.progress((i + 1) / len(bulk_images))
                    
                    with results_container:
                        rc1, rc2 = st.columns([1, 4])
                        with rc1:
                            st.image(img_pil, width=100, caption=f"Image {i+1}")
                        with rc2:
                            if seo_text_json:
                                seo_data = parse_json_response(seo_text_json)
                                if seo_data:
                                    with st.expander(f"✅ Tags for Image {i+1}", expanded=True):
                                        st.write("📄 **File Name:**")
                                        st.code(seo_data.get('file_name', 'N/A'), language="text")
                                        st.write("🏷️ **Alt Tag:**")
                                        st.code(seo_data.get('alt_tag', 'N/A'), language="text")
                                else:
                                    st.warning(f"Could not format JSON for Image {i+1}. Raw output:")
                                    st.code(seo_text_json)
                            else:
                                st.error(f"Failed image {i+1}: {error}")
                    st.divider()
                    time.sleep(0.5) 
            
            st.success("🎉 All images processed!")
            progress_bar.empty()

# === TAB 3: AI PRODUCT WRITER (NEW!) ===
with tab3:
    st.header("📝 AI SEO Product Content Writer")
    st.caption(f"Powered by {MODEL_TEXT_SEO} - Writes human-like, undetectable content.")
    
    col_w1, col_w2 = st.columns([1, 1.2])
    
    with col_w1:
        st.subheader("1. Product Input")
        
        # Image Upload for Context
        writer_img_file = st.file_uploader("Upload Product Image (Optional - for AI context)", type=["jpg", "png"], key="writer_img")
        writer_img = Image.open(writer_img_file) if writer_img_file else None
        if writer_img: st.image(writer_img, width=200, caption="Context Image")
        
        # SINGLE TEXT AREA FOR INPUT (UPDATED)
        st.markdown("👇 **Paste your Product Brief / Raw Data here:**")
        raw_input_data = st.text_area(
            "Paste Raw Details (URL, Keyword, Details, Dimensions...)", 
            height=300,
            placeholder="product url: ...\nคีย์เวิร์ดหลัก: ...\nคีย์เวิร์ดรอง: ...\nรายละเอียดสินค้า: ...",
            key="raw_input_box"
        )
        
        generate_content_btn = st.button("🚀 Generate Product Content", type="primary", use_container_width=True)
        
    with col_w2:
        st.subheader("2. Generated Content")
        
        if generate_content_btn:
            if not api_key:
                st.error("API Key required.")
            elif not raw_input_data:
                st.error("Please paste your product details in the text box.")
            else:
                with st.spinner("AI is analyzing data and writing content (10-20s)..."):
                    content_json, err = generate_product_content(api_key, writer_img, raw_input_data)
                    
                    if content_json:
                        data = parse_json_response(content_json)
                        if data:
                            # 1. Slug
                            st.write("🔗 **Recommended URL Slug:**")
                            st.code(data.get('url_slug', ''), language="text")
                            
                            # 2. Meta
                            st.write("🪪 **Meta Title (Max 60):**")
                            st.code(data.get('meta_title', ''), language="text")
                            st.write("📝 **Meta Description (Max 160):**")
                            st.code(data.get('meta_description', ''), language="text")
                            
                            # 3. H1
                            st.write("📌 **Product Title (H1):**")
                            st.code(data.get('product_title_h1', ''), language="text")
                            
                            # 4. Content
                            st.write("📄 **Product Description (HTML):**")
                            st.code(data.get('html_content', ''), language="html")
                            with st.expander("Preview HTML Content"):
                                st.markdown(data.get('html_content', ''), unsafe_allow_html=True)
                            
                            # 5. Image SEO
                            st.divider()
                            st.subheader("🖼️ Image SEO (8 Variations)")
                            img_seo_list = data.get('image_seo', [])
                            
                            for idx, item in enumerate(img_seo_list):
                                with st.container():
                                    cols = st.columns([0.5, 2, 2])
                                    if idx == 0 and writer_img:
                                        cols[0].image(writer_img, width=50)
                                    else:
                                        cols[0].write(f"#{idx+1}")
                                        
                                    cols[1].code(item.get('file_name', ''), language="text")
                                    cols[2].code(item.get('alt_tag', ''), language="text")
                        else:
                            st.error("Failed to parse JSON. Raw output:")
                            st.code(content_json)
                    else:
                        st.error(f"Generation Failed: {err}")

# === TAB 4: LIBRARY MANAGER ===
with tab4:
    st.subheader("🛠️ Prompt Library")
    target = st.session_state.edit_target
    form_title = f"✏️ Edit Style: {target['name']}" if target else "➕ Add New Style"
    
    with st.form("style_form"):
        st.write(f"**{form_title}**")
        c1, c2 = st.columns(2)
        n = c1.text_input("Name", value=target['name'] if target else "")
        c = c2.text_input("Category", value=target['category'] if target else "")
        t = st.text_area("Template", value=target['template'] if target else "A model wearing {color} ring...")
        v = st.text_input("Variables", value=target['variables'] if target else "color, size")
        u = st.text_input("Sample Image URL", value=target['sample_url'] if target else "")
        
        cols = st.columns([1, 4])
        submitted = cols[0].form_submit_button("💾 Save Style")
        if target:
            if cols[1].form_submit_button("❌ Cancel Edit"):
                st.session_state.edit_target = None
                st.rerun()

        if submitted:
            new_data = {
                "id": target['id'] if target else str(len(st.session_state.library) + 1000),
                "name": n, "category": c, "template": t, "variables": v, "sample_url": u
            }
            if target:
                for idx, item in enumerate(st.session_state.library):
                    if item['id'] == target['id']:
                        st.session_state.library[idx] = new_data
                        break
                st.success("Updated!")
            else:
                st.session_state.library.append(new_data)
                st.success("Added!")
            save_prompts(st.session_state.library)
            st.session_state.edit_target = None
            st.rerun()

    st.divider()
    for i, p in enumerate(st.session_state.library):
        c1, c2, c3, c4 = st.columns([1, 4, 1, 1])
        # Safe Image Loading
        try:
            if p.get("sample_url"): st.image(p["sample_url"], width=50)
        except: st.caption("No Preview")
        
        c2.write(f"**{p['name']}** ({p['category']})")
        if c3.button("✏️ Edit", key=f"edit_{i}"):
            st.session_state.edit_target = p
            st.rerun()
        if c4.button("🗑️ Del", key=f"del_{i}"):
            st.session_state.library.pop(i)
            save_prompts(st.session_state.library)
            st.rerun()

# === TAB 5: ABOUT MODELS ===
with tab5:
    st.header("🔍 Check Gemini Model Availability")
    if st.button("📡 Scan Available Models"):
        if not api_key:
            st.error("Please enter your API Key in the sidebar settings first.")
        else:
            with st.spinner("Connecting to Google API..."):
                models_data = list_available_models(api_key)
                if models_data:
                    gemini_models = []
                    for m in models_data:
                        if "gemini" in m['name']:
                            gemini_models.append({
                                "Model ID": m['name'],
                                "Version": m['version'],
                                "Display Name": m['displayName']
                            })
                    if gemini_models:
                        st.success(f"Found {len(gemini_models)} Gemini models!")
                        st.dataframe(pd.DataFrame(gemini_models), use_container_width=True)
                        st.info(f"Current Config:\n- Image Gen: `{MODEL_IMAGE_GEN}`\n- SEO Text: `{MODEL_TEXT_SEO}`")
                    else:
                        st.warning("No Gemini models found.")
                else:
                    st.error("Failed to fetch models. Please check your API Key validity.")
