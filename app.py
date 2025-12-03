import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Jewelry AI Studio")

# Load API Key from Streamlit Secrets (จะสอนตั้งค่าในขั้นตอนต่อไป)
# หรือถ้าเทสในเครื่อง ให้ใส่ key ตรงๆ ไปก่อนตรงนี้: api_key = "YOUR_KEY"
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = st.text_input("Enter Gemini API Key", type="password")

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}"

# --- 2. FUNCTIONS ---
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_image(input_image, prompt_text, category):
    if not api_key:
        st.error("Please enter an API Key.")
        return None

    base64_img = image_to_base64(input_image)
    
    # Prompt Engineering for Jewelry
    full_prompt = f"""
    Task: Virtual Try-On for Jewelry.
    Category: {category}
    Style/Instruction: {prompt_text}
    
    IMPORTANT: 
    1. Keep the product (jewelry) in the input image EXACTLY as it is. Do not change its design.
    2. Generate a realistic human model wearing this product naturally.
    3. High luxury fashion photography style.
    """

    # Payload for Gemini
    payload = {
        "contents": [{
            "parts": [
                {"text": full_prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_img
                    }
                }
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
        response = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# --- 3. UI LAYOUT ---
st.title("💎 Jewelry AI Studio")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Input Product")
    
    # Streamlit รับรูปได้ทั้ง Upload และ Ctrl+V (ผ่าน Browse files)
    uploaded_file = st.file_uploader("Upload or Drag & Drop Product Image", type=["jpg", "png", "jpeg"])
    
    product_image = None
    if uploaded_file is not None:
        product_image = Image.open(uploaded_file)
        st.image(product_image, caption="Original Product", use_column_width=True)

with col2:
    st.header("2. Settings & Generate")
    
    category = st.selectbox("Product Category", ["Ring (แหวน)", "Necklace (สร้อยคอ)", "Pendant (จี้)", "Wallet Chain"])
    
    # Preset Prompts
    style_option = st.selectbox("Choose Style", [
        "Custom (พิมพ์เอง)",
        "Luxury Hand Model (Studio Light)",
        "Streetwear Vibe (Outdoor)",
        "Minimalist Skin Tone"
    ])
    
    prompt_input = ""
    if style_option == "Luxury Hand Model (Studio Light)":
        prompt_input = "A realistic close-up of a female hand model wearing this ring. Soft studio lighting, elegant background, 8k resolution."
    elif style_option == "Streetwear Vibe (Outdoor)":
        prompt_input = "A fashion shot of a model wearing this item, streetwear outfit, urban city background, natural lighting."
    elif style_option == "Minimalist Skin Tone":
        prompt_input = "Clean minimal shot, focus on the jewelry on skin, beige tone background."
    
    user_prompt = st.text_area("Prompt / Instruction", value=prompt_input, height=150)
    
    generate_btn = st.button("✨ GENERATE IMAGE", type="primary", use_container_width=True)

# --- 4. RESULT AREA ---
if generate_btn and product_image:
    with st.spinner("AI is working on your jewelry... (Takes 10-20s)"):
        result = generate_image(product_image, user_prompt, category)
        
        if result:
            # Handle Gemini Response
            try:
                # กรณี Gemini ส่งกลับเป็น Base64
                if "candidates" in result:
                    content = result["candidates"][0]["content"]["parts"][0]
                    if "inline_data" in content:
                        img_data = base64.b64decode(content["inline_data"]["data"])
                        st.success("Generation Complete!")
                        st.image(img_data, caption="Generated Result", use_column_width=True)
                        
                        # Download Button
                        st.download_button(
                            label="Download Image",
                            data=img_data,
                            file_name="jewelry_gen.jpg",
                            mime="image/jpeg"
                        )
                    else:
                        st.warning("Model replied with text instead of image (Check Prompt).")
                        st.write(content.get("text", ""))
            except Exception as e:
                st.error(f"Error parsing result: {e}")