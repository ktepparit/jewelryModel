import streamlit as st
import json
import os
import requests
import base64
from io import BytesIO
from PIL import Image

# --- 1. SETUP & CONFIG ---
st.set_page_config(layout="wide", page_title="Jewelry AI Studio", page_icon="💎")

# Default Prompts
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

# --- 2. DATABASE FUNCTIONS ---
def load_prompts():
    if not os.path.exists(PROMPT_DB_FILE):
        try:
            with open(PROMPT_DB_FILE, "w") as f:
                json.dump(DEFAULT_PROMPTS, f)
            return DEFAULT_PROMPTS
        except:
            return DEFAULT_PROMPTS
    try:
        with open(PROMPT_DB_FILE, "r") as f:
            return json.load(f)
    except:
        return DEFAULT_PROMPTS

def save_prompts(prompts_data):
    try:
        with open(PROMPT_DB_FILE, "w") as f:
            json.dump(prompts_data, f)
    except:
        pass

if "prompt_library" not in st.session_state:
    st.session_state.prompt_library = load_prompts()

# --- 3. HELPER FUNCTIONS ---
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def call_gemini_api(api_key, image, prompt):
    # --- จุดที่แก้ไข: เปลี่ยน Model ID เป็น gemini-3-pro-image-preview ---
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key={api_key}"
    
    # ** ถ้าใช้ตัวบนแล้วยัง 404 ให้ลองเอา # ออกจากบรรทัดล่างนี้เพื่อใช้ตัวเสถียรแทนครับ **
    # api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
    
    base64_img = image_to_base64(image)
    
    payload = {
        "contents": [{
            "parts": [
                # เพิ่ม Instruction ให้ชัดเจนขึ้นสำหรับ Gemini 3
                {"text": f"Task: AI Virtual Try-on product photography. \nInstruction: {prompt} \nConstraint: The jewelry product in the image MUST remain exactly as it is in the input. Do not alter the jewelry design. Generate a realistic human model wearing it."},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_img}}
            ]
        }],
        "generationConfig": {
            # Gemini 3 อาจต้องการ temperature ที่ต่ำลงเพื่อความแม่นยำ
            "temperature": 0.3,
            "topK": 32,
            "topP": 0.95,
            "maxOutputTokens": 2048
        }
    }
    
    try:
        response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and result["candidates"]:
                content = result["candidates"][0]["content"]["parts"][0]
                if "inline_data" in content:
                    return base64.b64decode(content["inline_data"]["data"]), None
                elif "text" in content:
                    return None, "Model returned text: " + content["text"]
            return None, f"Unknown response format: {result}"
        else:
            # คืนค่า Error เต็มๆ เพื่อให้ Debug ง่ายขึ้น
            return None, f"API Error: {response.text}"
    except Exception as e:
        return None, str(e)

# --- 4. MAIN UI ---
with st.sidebar:
    st.title("⚙️ Settings")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ API Key Loaded")
    except:
        api_key = st.text_input("Enter Gemini API Key", type="password")
    st.info("Tip: Use 'Manage Library' to add custom prompts.")

st.title("💎 Jewelry AI Studio")

tab_gen, tab_manager = st.tabs(["✨ Create Image", "📚 Manage Library"])

# === TAB 1: GENERATE IMAGE ===
with tab_gen:
    col_input, col_config = st.columns([1, 1.2])

    with col_input:
        st.subheader("1. Input Product")
        uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])
        final_image = None
        if uploaded_file:
            final_image = Image.open(uploaded_file)
            st.image(final_image, caption="Product Preview", use_column_width=True)
        else:
            st.info("👆 Please upload an image first.")

    with col_config:
        st.subheader("2. Select Style & Edit")
        
        if st.session_state.prompt_library:
            all_prompts = st.session_state.prompt_library
            categories = list(set([p['category'] for p in all_prompts]))
            selected_cat = st.selectbox("Category", categories)
            
            filtered_prompts = [p for p in all_prompts if p['category'] == selected_cat]
            
            if filtered_prompts:
                selected_style_id = st.radio(
                    "Choose Style:",
                    options=[p['id'] for p in filtered_prompts],
                    format_func=lambda x: next(p['name'] for p in filtered_prompts if p['id'] == x)
                )
                
                current_style = next(p for p in filtered_prompts if p['id'] == selected_style_id)
                
                if current_style.get("sample_url"):
                    st.image(current_style["sample_url"], width=15
