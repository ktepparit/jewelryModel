import streamlit as st
import json
import os
import requests
import base64
from io import BytesIO
from PIL import Image

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
        "sample_url": "https://images.unsplash.com/photo-1599643478518-17488fbbcd75?q=80&w=300&auto=format&fit=crop"
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
        with open(PROMPT_DB_FILE, "w") as f:
            json.dump(DEFAULT_PROMPTS, f)
        return DEFAULT_PROMPTS
    try:
        with open(PROMPT_DB_FILE, "r") as f:
            return json.load(f)
    except:
        return DEFAULT_PROMPTS

def save_prompts(prompts_data):
    with open(PROMPT_DB_FILE, "w") as f:
        json.dump(prompts_data, f)

# Load data into Session State
if "prompt_library" not in st.session_state:
    st.session_state.prompt_library = load_prompts()

# --- 3. HELPER FUNCTIONS ---
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def call_gemini_api(api_key, image, prompt):
    # ใช้ Endpoint มาตรฐานสำหรับ Image Understanding/Generation
    # หมายเหตุ: Model ID อาจต้องปรับเปลี่ยนตาม Provider ที่คุณใช้
    # ถ้าใช้ Google AI Studio โดยตรงแนะนำให้ใช้ gemini-1.5-pro หรือ gemini-1.5-flash
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    
    base64_img = image_to_base64(image)
    
    payload = {
        "contents": [{
            "parts": [
                {"text": f"Task: You are an AI Image Generator. \nInstruction: {prompt} \nConstraint: Use the jewelry in the input image exactly as is. Generate a realistic look."},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_img}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.4,
            "topK": 32,
            "topP": 1,
            "maxOutputTokens": 2048
        }
    }
    
    try:
        response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result:
                content = result["candidates"][0]["content"]["parts"][0]
                
                # กรณี 1: ตอบกลับเป็นรูป (Base64)
                if "inline_data" in content:
                    return base64.b64decode(content["inline_data"]["data"]), None
                
                # กรณี 2: ตอบกลับเป็น Text (บางที Gemini จะอธิบายภาพแทนถ้าโมเดลยังไม่รองรับการ Gen รูปโดยตรง)
                elif "text" in content:
                    return None, "Model returned text instead of image (Check Model ID): " + content["text"]
            
            return None, "Unknown response format"
        else:
            return None, f"API Error: {response.text}"
    except Exception as e:
        return None, str(e)

# --- 4. MAIN UI ---

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ API Key Loaded")
    except:
        api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.info("Tip: Use 'Manage Library' to add custom prompts.")

st.title("💎 Jewelry AI Studio")

# Tabs
tab_gen, tab_manager = st.tabs(["✨ Create Image", "📚 Manage Library"])

# === TAB 1: GENERATE IMAGE ===
with tab_gen:
    col_input, col_config = st.columns([1, 1.2])

    with col_input:
        st.subheader("1. Input Product")
        # ใช้ File Uploader ธรรมดา (เสถียรที่สุด)
        uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])
        
        final_image = None
        if uploaded_file:
            final_image = Image.open(uploaded_file)
            st.image(final_image, caption="Product Preview", use_column_width=True)
        else:
            st.info("👆 Please upload an image to start.")

    with col_config:
        st.subheader("2. Select Style")
        
        if st.session_state.prompt_library:
            all_prompts = st.session_state.prompt_library
            categories = list(set([p['category'] for p in all_prompts]))
            selected_cat = st.selectbox("Category", categories)
            
            # Filter
            filtered_prompts = [p for p in all_prompts if p['category'] == selected_cat]
            
            if filtered_prompts:
                # Style Selection
                selected_style_id = st.radio(
                    "Choose Style:",
                    options=[p['id'] for p in filtered_prompts],
                    format_func=lambda x: next(p['name'] for p in filtered_prompts if p['id'] == x)
                )
                
                current_style = next(p for p in filtered_prompts if p['id'] == selected_style_id)
                
                # Thumbnail
                if current_style.get("sample_url"):
                    st.image(current_style["sample_url"], width=200)
                
                st.markdown("---")
                st.write("📝 **Parameters**")
                
                # Dynamic Vars
                variables = [v.strip() for v in current_style.get('variables', '').split(",")]
                user_vars = {}
                for var in variables:
                    if var:
                        user_vars[var] = st.text_input(f"Value for '{var}'", placeholder="e.g. large, gold")
                
                # Build Prompt
                final_prompt_text = current_style['template']
                for var, val in user_vars.items():
                    final_prompt_text = final_prompt_text.replace(f"{{{var}}}", val if val else "")
                
                with st.expander("View Final Prompt"):
                    st.code(final_prompt_text)
                
                # Generate Button
                if st.button("🚀 GENERATE IMAGE", type="primary", use_container_width=True):
                    if not final_image or not api_key:
                        st.error("Missing Image or API Key")
                    else:
                        with st.spinner("AI is generating... (Please wait)"):
                            img_data, error = call_gemini_api(api_key, final_image, final_prompt_text)
                            if img_data:
                                st.balloons()
                                st.success("Done!")
                                st.image(img_data, use_column_width=True)
                                st.download_button("Download Image", img_data, "jewelry_ai_gen.jpg", "image/jpeg")
                            else:
                                st.error(error)
            else:
                st.warning("No styles found for this category.")
        else:
            st.warning("Prompt Library is empty.")

# === TAB 2: MANAGE LIBRARY ===
with tab_manager:
    st.subheader("🛠️ Prompt Manager")
    
    with st.expander("➕ Add New Style", expanded=False):
        with st.form("add_prompt"):
            new_name = st.text_input("Style Name")
            new_cat = st.text_input("Category (e.g. Ring, Pendant)")
            new_temp = st.text_area("Template (Use {var} for variables)", "A model wearing {color} ring...")
            new_vars = st.text_input("Variables (comma separated)", "color, size")
            new_url = st.text_input("Sample Image URL")
            
            if st.form_submit_button("Save Style"):
                new_entry = {
                    "id": f"p{len(st.session_state.prompt_library) + 100}",
                    "name": new_name,
                    "category": new_cat,
                    "template": new_temp,
                    "variables": new_vars,
                    "sample_url": new_url
                }
                st.session_state.prompt_library.append(new_entry)
                save_prompts(st.session_state.prompt_library)
                st.success("Saved!")
                st.rerun()
    
    st.divider()
    for idx, p in enumerate(st.session_state.prompt_library):
        c1, c2, c3 = st.columns([1, 4, 1])
        with c1:
            if p.get("sample_url"):
                st.image(p["sample_url"], width=50)
        with c2:
            st.write(f"**{p['name']}** - {p['category']}")
        with c3:
            if st.button("Delete", key=f"del_{idx}"):
                st.session_state.prompt_library.pop(idx)
                save_prompts(st.session_state.prompt_library)
                st.rerun()
