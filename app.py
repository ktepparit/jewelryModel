import streamlit as st
import json
import requests
import base64
from io import BytesIO
from PIL import Image
import time
import pandas as pd # เพิ่ม pandas เพื่อแสดงตารางสวยๆ

# --- 1. CONFIGURATION & CONSTANTS ---
st.set_page_config(layout="wide", page_title="Jewelry AI Studio")

# Prompt A: สำหรับ Gen SEO หลังสร้างรูปเสร็จ (Tab 1)
SEO_PROMPT_POST_GEN = """
You are an SEO specialist with 15-20 years of experience. 
Help write SEO-optimized image file name with image alt tags in English for the product image with a model created, having product details according to this url: {product_url}
To rank well on organic search engines by customer groups interested in this type of product.

Please provide the output exactly in this format:
---
File Name: [your-optimized-filename.jpg]
Alt Tag: [Your optimized descriptive alt tag with keywords]
---
"""

# Prompt B: สำหรับ Bulk SEO รูปที่มีอยู่แล้ว (Tab 2)
SEO_PROMPT_BULK_EXISTING = """
คุณคือ SEO specialist ที่มีประสบการณ์ 15-20 ปี ช่วยเขียน SEO-optimized image file name with image alt tags เป็นภาษาอังกฤษ สำหรับสินค้าของฉันตามแต่ละรูปที่แนบมาให้ โดยมีรายละเอียดของสินค้าตาม url นี้ {product_url} เพื่อให้ได้ติดอันดับที่ดีบน organic search engine โดยกลุ่มลูกค้าเป็นผู้สนใจสินค้าชนิดนี้

Please provide the output clearly separating File Name and Alt Tag.
"""

# Default Data
DEFAULT_PROMPTS = [
    {
        "id": "p1", "name": "Luxury Hand (Ring)", "category": "Ring",
        "template": "A realistic close-up of a female hand model wearing a ring with {face_size} face size, soft studio lighting, elegant jewelry photography.",
        "variables": "face_size",
        "sample_url": "https://images.unsplash.com/photo-1599643478518-17488fbbcd75?q=80&w=300&auto=format&fit=crop"
    },
    {
        "id": "p2", "name": "Streetwear Necklace", "category": "Necklace",
        "template": "A fashion portrait of a model wearing a {length} necklace, streetwear outfit, urban background, high detailed texture.",
        "variables": "length",
        "sample_url": "https://images.unsplash.com/photo-1611591437281-460bfbe1220a?q=80&w=300&auto=format&fit=crop"
    }
]

# --- 2. CLOUD DATABASE FUNCTIONS (JsonBin.io) ---
def get_prompts():
    try:
        API_KEY = st.secrets["JSONBIN_API_KEY"]
        BIN_ID = st.secrets["JSONBIN_BIN_ID"]
        url = f"https://api.jsonbin.io/v3/b/{BIN_ID}/latest"
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
        url = f"https://api.jsonbin.io/v3/b/{BIN_ID}"
        headers = {"Content-Type": "application/json", "X-Master-Key": API_KEY}
        requests.put(url, json=data, headers=headers)
    except Exception as e:
        st.error(f"Save failed: {e}")

# --- 3. HELPER FUNCTIONS (AI & Image) ---
def img_to_base64(img):
    buf = BytesIO()
    if img.mode == 'RGBA': img = img.convert('RGB')
    img.thumbnail((1024, 1024)) 
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

# ฟังก์ชัน Gen รูปภาพ
def generate_image(api_key, image_list, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key={api_key}"
    parts = [{"text": f"Instruction: {prompt} \nConstraint: Keep the jewelry products in the input images EXACTLY as they are. Analyze all images to understand the 3D structure. Generate a realistic model wearing it."}]
    for img in image_list:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})
    try:
        res = requests.post(url, json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.3}}, headers={"Content-Type": "application/json"})
        if res.status_code != 200: return None, f"API Error: {res.text}"
        content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
        if "inline_data" in content: return base64.b64decode(content["inline_data"]["data"]), None
        if "inlineData" in content: return base64.b64decode(content["inlineData"]["data"]), None
        if "text" in content: return None, f"Model returned text: {content['text']}"
        return None, "Unknown response format."
    except Exception as e: return None, str(e)

# ฟังก์ชัน Gen SEO (Tab 1)
def generate_seo_tags_post_gen(api_key, product_url):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent?key={api_key}"
    final_seo_prompt = SEO_PROMPT_POST_GEN.replace("{product_url}", product_url)
    payload = {
        "contents": [{"parts": [{"text": final_seo_prompt}]}],
        "generationConfig": {"temperature": 0.7}
    }
    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if res.status_code != 200: return None, f"API Error: {res.text}"
        content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
        if "text" in content: return content["text"], None
        return None, "No text returned from model."
    except Exception as e: return None, str(e)

# ฟังก์ชัน Bulk SEO (Tab 2)
def generate_seo_for_existing_image(api_key, img_pil, product_url):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent?key={api_key}"
    final_prompt = SEO_PROMPT_BULK_EXISTING.replace("{product_url}", product_url)
    payload = {
        "contents": [{
            "parts": [
                {"text": final_prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img_pil)}}
            ]
        }],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 2048}
    }
    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if res.status_code != 200: return None, f"API Error: {res.text}"
        content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
        if "text" in content: return content["text"], None
        return None, "Model returned no text."
    except Exception as e: return None, str(e)

# --- NEW FUNCTION: Check Available Models (Tab 4) ---
def list_available_models(api_key):
    # Endpoint มาตรฐานสำหรับเช็คโมเดลทั้งหมดที่ Key นี้เข้าถึงได้
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
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
if "image_generated_success" not in st.session_state:
    st.session_state.image_generated_success = False

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
        st.warning("⚠️ Local Mode")

st.title("💎 Jewelry AI Studio")

# เพิ่ม Tab 4 เข้าไป
tab1, tab2, tab3, tab4 = st.tabs(["✨ Generate Image", "🏷️ Bulk SEO Tags", "📚 Library Manager", "ℹ️ About Models"])

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
            if sel_style.get("sample_url"): st.image(sel_style["sample_url"], width=100)
            
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
                else:
                    with st.spinner("Generating Image (Gemini 3 Pro)..."):
                        d, e = generate_image(api_key, images_to_send, prompt_edit)
                        if d:
                            st.image(d)
                            st.download_button("Download", d, "gen.jpg")
                            st.session_state.image_generated_success = True 
                        else:
                            st.error(e)
                            st.session_state.image_generated_success = False

            if st.session_state.image_generated_success:
                st.divider()
                st.subheader("🌍 SEO Tools (Post-Generation)")
                st.caption("Generate tags for the NEW image above.")
                product_url_input = st.text_input("Paste Product URL here:", placeholder="https://yourshop.com/product/...", key="post_gen_url")
                
                if st.button("✨ Gen Tags for New Image"):
                    if not product_url_input:
                        st.warning("Please enter a Product URL first.")
                    else:
                        with st.spinner("Consulting SEO Specialist AI (Gemini 3 Pro)..."):
                            seo_result, seo_err = generate_seo_tags_post_gen(api_key, product_url_input)
                            if seo_result:
                                with st.expander("✅ SEO Tags Generated!", expanded=True):
                                    st.code(seo_result, language="yaml")
                            else:
                                st.error(f"SEO Generation Failed: {seo_err}")

        else: st.warning("Library empty.")

# === TAB 2: BULK SEO TAGS ===
with tab2:
    st.header("🏷️ Generate SEO Tags for Existing Images")
    st.caption("อัปโหลดรูปภาพสินค้าที่มีอยู่แล้ว เพื่อให้ AI ช่วยเขียน File Name และ Alt Tag ตาม URL สินค้า")
    
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
        bulk_url = st.text_input("Product URL (ใช้เป็นข้อมูลอ้างอิงสำหรับทุกรูป):", placeholder="https://yourshop.com/product/...", key="bulk_seo_url")
        
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
                with st.spinner(f"Analyzing Image {i+1}/{len(bulk_images)} with Gemini 3 Pro..."):
                    seo_text, error = generate_seo_for_existing_image(api_key, img_pil, bulk_url)
                    progress_bar.progress((i + 1) / len(bulk_images))
                    
                    with results_container:
                        rc1, rc2 = st.columns([1, 4])
                        with rc1:
                            st.image(img_pil, width=100, caption=f"Image {i+1}")
                        with rc2:
                            if seo_text:
                                with st.expander(f"✅ Tags for Image {i+1}", expanded=True):
                                    st.code(seo_text, language="markdown") 
                            else:
                                st.error(f"Failed image {i+1}: {error}")
                    st.divider()
                    time.sleep(0.5) 
            
            st.success("🎉 All images processed!")
            progress_bar.empty()

# === TAB 3: LIBRARY MANAGER ===
with tab3:
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
        if p.get("sample_url"): c1.image(p["sample_url"], width=50)
        c2.write(f"**{p['name']}** ({p['category']})")
        if c3.button("✏️ Edit", key=f"edit_{i}"):
            st.session_state.edit_target = p
            st.rerun()
        if c4.button("🗑️ Del", key=f"del_{i}"):
            st.session_state.library.pop(i)
            save_prompts(st.session_state.library)
            st.rerun()

# === NEW TAB 4: ABOUT MODELS (ตรวจสอบสิทธิ์) ===
with tab4:
    st.header("🔍 Check Gemini Model Availability")
    st.write("หน้านี้จะช่วยตรวจสอบว่า API Key ของคุณสามารถเข้าถึงโมเดลใดได้บ้างจาก Google Server โดยตรง")
    
    if st.button("📡 Scan Available Models"):
        if not api_key:
            st.error("Please enter your API Key in the sidebar settings first.")
        else:
            with st.spinner("Connecting to Google API..."):
                models_data = list_available_models(api_key)
                
                if models_data:
                    # เตรียมข้อมูลสำหรับแสดงผล
                    gemini_models = []
                    for m in models_data:
                        # กรองเอาเฉพาะโมเดล Gemini เพื่อความดูง่าย (หรือจะเอาทั้งหมดก็ได้)
                        if "gemini" in m['name']:
                            gemini_models.append({
                                "Model ID (Name)": m['name'],
                                "Version": m['version'],
                                "Display Name": m['displayName'],
                                "Input Token Limit": m.get('inputTokenLimit', 'N/A'),
                                "Methods": ", ".join(m.get('supportedGenerationMethods', []))
                            })
                    
                    if gemini_models:
                        st.success(f"Found {len(gemini_models)} Gemini models accessible by your key!")
                        
                        # สร้าง DataFrame เพื่อแสดงเป็นตารางสวยงาม
                        df = pd.DataFrame(gemini_models)
                        st.dataframe(df, use_container_width=True)
                        
                        st.info("💡 **Tip:** Look for models like `gemini-3-pro-preview` or `gemini-1.5-flash` in the list. If they appear here, the app can use them.")
                    else:
                        st.warning("Connected successfully, but no 'Gemini' models were found in the list. You might only have access to PaLM legacy models.")
                        st.write(models_data) # Show raw data just in case
                else:
                    st.error("Failed to fetch models. Please check your API Key validity.")

