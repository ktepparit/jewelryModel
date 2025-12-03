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
    # ใช้ Model ID: gemini-1.5-flash เพื่อความเร็วและประหยัด (หรือเปลี่ยนเป็น pro ได้)
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    
    base64_img = image_to_base64(image)
    
    payload = {
        "contents": [{
            "parts": [
                {"text": f"Task: AI Image Generator / Virtual Try-on. \nInstruction: {prompt} \nConstraint: Keep the jewelry product in the input image EXACTLY as is. Do not hallucinate new jewelry designs. Make it look realistic."},
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
                if "inline_data" in content:
                    return base64.b64decode(content["inline_data"]["data"]), None
                elif "text" in content:
                    return None, "Model returned text: " + content["text"]
            return None, "Unknown response format"
        else:
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
        st.
