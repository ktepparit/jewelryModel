import streamlit as st
import json
import os
import requests
import base64
from io import BytesIO
from PIL import Image
from streamlit_paste_image import paste_image # หัวใจสำคัญของการ Paste

# --- 1. SETUP & CONFIG ---
st.set_page_config(layout="wide", page_title="Jewelry AI Studio", page_icon="💎")

# Default Prompts (ถ้ายังไม่มีไฟล์ Database)
DEFAULT_PROMPTS = [
    {
        "id": "p1",
        "name": "Luxury Hand (Ring)",
        "category": "Ring",
        "template": "A realistic close-up of a female hand model wearing a ring with {face_size} face size, soft studio lighting, elegant jewelry photography.",
        "variables": "face_size",
        "sample_url": "https://images.unsplash.com/photo-1599643478518-17488fbbcd75?q=80&w=300&auto=format&fit=crop" # รูปตัวอย่าง
    },
    {
        "id": "p2",
        "name": "Streetwear Necklace",
        "category": "Necklace",
        "template": "A fashion portrait of a model wearing a {length} necklace, streetwear outfit, urban background, high detailed texture.",
        "variables": "length",
        "sample_url": "https://images.unsplash.com/photo-1611591437281-460bfbe1220a?q=80&w=300&auto=format&fit=crop"
    }
]

PROMPT_DB_FILE = "prompt_library.json"

# --- 2. DATABASE FUNCTIONS (JSON) ---
def load_prompts():
    if not os.path.exists(PROMPT_DB_FILE):
        # ถ้าไม่มีไฟล์ ให้สร้างใหม่จาก Default
        with open(PROMPT_DB_FILE, "w") as f:
            json.dump(DEFAULT_PROMPTS, f)
        return DEFAULT_PROMPTS
    with open(PROMPT_DB_FILE, "r") as f:
        return json.load(f)

def save_prompts(prompts_data):
    with open(PROMPT_DB_FILE, "w") as f:
        json.dump(prompts_data, f)

# Load data เข้า Session State เพื่อลดการอ่านไฟล์ซ้ำ
if "prompt_library" not in st.session_state:
    st.session_state.prompt_library = load_prompts()

# --- 3. HELPER FUNCTIONS ---
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def call_gemini_api(api_key, image, prompt):
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}"
    base64_img = image_to_base64(image)
    
    payload = {
        "contents": [{
            "parts": [
                {"text": f"Task: Virtual Try-On Jewelry. \nConstraint: Keep product EXACTLY as is. \nInstruction: {prompt}"},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_img}}
            ]
        }],
        "generationConfig": {"temperature": 0.4, "topK": 32, "topP": 1, "maxOutputTokens": 2048}
    }
    try:
        response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result:
                content = result["candidates"][0]["content"]["parts"][0]
                if "inline_data" in content:
                    return base64.b64decode(content["inline_data"]["data"]), None
                elif "text" in content:
                    return None, "Model returned text: " + content["text"]
        return None, f"API Error: {response.text}"
    except Exception as e:
        return None, str(e)

# --- 4. MAIN UI ---

# Sidebar for Key
with st.sidebar:
    st.title("⚙️ Settings")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("API Key Loaded from Secrets")
    except:
        api_key = st.text_input("Gemini API Key", type="password")
    
    st.info("💡 **Tip:** Use 'Manage Library' tab to add your own prompts and styles.")

st.title("💎 Jewelry AI Studio Pro")

# แบ่ง Tab การทำงาน
tab_gen, tab_manager = st.tabs(["✨ Create Image", "📚 Manage Library"])

# === TAB 1: GENERATE IMAGE ===
with tab_gen:
    col_input, col_config = st.columns([1, 1.2])

    with col_input:
        st.subheader("1. Input Product")
        st.markdown("วางรูปสินค้าของคุณที่นี่ (Paste Image)")
        
        # --- PASTE FEATURE ---
        paste_result = paste_image() # ปุ่ม Paste อยู่ตรงนี้
        
        final_image = None
        
        if paste_result.image_data is not None:
            st.success("Image Pasted!")
            final_image = paste_result.image_data
            st.image(final_image, caption="Pasted Product", use_column_width=True)
        else:
            # Fallback Uploader
            uploaded_file = st.file_uploader("Or upload file", type=["jpg", "png"])
            if uploaded_file:
                final_image = Image.open(uploaded_file)
                st.image(final_image, caption="Uploaded Product", use_column_width=True)

    with col_config:
        st.subheader("2. Select Style & Config")
        
        # กรอง Category
        all_prompts = st.session_state.prompt_library
        categories = list(set([p['category'] for p in all_prompts]))
        selected_cat = st.selectbox("Category", categories)
        
        # Filter Prompts by Category
        filtered_prompts = [p for p in all_prompts if p['category'] == selected_cat]
        
        # Show Prompts as Grid/Cards
        selected_style_id = st.radio(
            "Choose a Style:",
            options=[p['id'] for p in filtered_prompts],
            format_func=lambda x: next(p['name'] for p in filtered_prompts if p['id'] == x)
        )
        
        # Get Selected Prompt Data
        current_style = next(p for p in filtered_prompts if p['id'] == selected_style_id)
        
        # Show Thumbnail
        if current_style.get("sample_url"):
            st.image(current_style["sample_url"], width=200, caption="Style Preview")
        
        st.markdown("---")
        
        # Dynamic Variables Input
        st.write("📝 **Parameters**")
        variables = [v.strip() for v in current_style['variables'].split(",")]
        user_vars = {}
        
        for var in variables:
            if var:
                user_vars[var] = st.text_input(f"Value for '{var}'", placeholder=f"e.g. large, 18 inch")
        
        # Final Prompt Construction
        final_prompt_text = current_style['template']
        for var, val in user_vars.items():
            final_prompt_text = final_prompt_text.replace(f"{{{var}}}", val if val else "")
            
        with st.expander("Preview Final Prompt"):
            st.code(final_prompt_text)
            
        # Generate Button
        if st.button("🚀 GENERATE IMAGE", type="primary", use_container_width=True):
            if not final_image or not api_key:
                st.error("Please provide Image and API Key")
            else:
                with st.spinner("Processing..."):
                    img_data, error = call_gemini_api(api_key, final_image, final_prompt_text)
                    if img_data:
                        st.balloons()
                        st.subheader("Result")
                        st.image(img_data, use_column_width=True)
                        st.download_button("Download", img_data, "gen_jewelry.jpg", "image/jpeg")
                    else:
                        st.error(error)

# === TAB 2: MANAGE LIBRARY ===
with tab_manager:
    st.subheader("🛠️ Prompt Library Manager")
    
    # 1. Add New Prompt
    with st.expander("➕ Add New Style", expanded=False):
        with st.form("add_prompt_form"):
            new_name = st.text_input("Style Name (e.g. Ring Luxury)")
            new_cat = st.text_input("Category (e.g. Ring)")
            new_template = st.text_area("Prompt Template (Use {variable} for dynamic parts)", 
                                        "A photo of a model wearing {size} ring...")
            new_vars = st.text_input("Variables (comma separated)", "size, color")
            new_img_url = st.text_input("Sample Image URL (Optional)")
            
            submitted = st.form_submit_button("Save Style")
            if submitted:
                new_entry = {
                    "id": f"p{len(st.session_state.prompt_library) + 100}", # Simple ID gen
                    "name": new_name,
                    "category": new_cat,
                    "template": new_template,
                    "variables": new_vars,
                    "sample_url": new_img_url
                }
                st.session_state.prompt_library.append(new_entry)
                save_prompts(st.session_state.prompt_library)
                st.success("Saved! Please refresh or go back to Create Image tab.")
                st.rerun()

    # 2. List & Delete
    st.write("### Your Saved Styles")
    for idx, p in enumerate(st.session_state.prompt_library):
        c1, c2, c3 = st.columns([1, 3, 1])
        with c1:
            if p.get("sample_url"):
                st.image(p["sample_url"], width=60)
            else:
                st.write("No Img")
        with c2:
            st.write(f"**{p['name']}** ({p['category']})")
            st.caption(f"Vars: {p['variables']}")
        with c3:
            if st.button("Delete", key=f"del_{p['id']}"):
                st.session_state.prompt_library.pop(idx)
                save_prompts(st.session_state.prompt_library)
                st.rerun()
        st.divider()
