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
                    st.image(current_style["sample_url"], width=150)
                
                st.markdown("---")
                
                # Dynamic Variables
                variables = [v.strip() for v in current_style.get('variables', '').split(",")]
                user_vars = {}
                if variables and variables[0] != "":
                    st.write("📝 **Parameters**")
                    for var in variables:
                        user_vars[var] = st.text_input(f"Value for '{var}'", placeholder="e.g. large, gold")
                
                # --- ส่วนสำคัญ: คำนวณ Prompt และเปิดให้แก้ไข ---
                base_prompt = current_style['template']
                for var, val in user_vars.items():
                    base_prompt = base_prompt.replace(f"{{{var}}}", val if val else "")
                
                st.write("✏️ **Final Prompt (Editable):**")
                # ใช้ text_area แทน code เพื่อให้ user แก้ไขข้อความได้ก่อนส่ง
                final_prompt_editable = st.text_area(
                    "You can edit the prompt below before generating:", 
                    value=base_prompt, 
                    height=120
                )
                
                if st.button("🚀 GENERATE IMAGE", type="primary", use_container_width=True):
                    if not final_image or not api_key:
                        st.error("Missing Image or API Key")
                    else:
                        with st.spinner("AI is generating..."):
                            # ส่ง final_prompt_editable ไปแทน
                            img_data, error = call_gemini_api(api_key, final_image, final_prompt_editable)
                            if img_data:
                                st.balloons()
                                st.success("Done!")
                                st.image(img_data, use_column_width=True)
                                st.download_button("Download", img_data, "gen_jewelry.jpg", "image/jpeg")
                            else:
                                st.error(error)
            else:
                st.warning("No styles found.")
        else:
            st.warning("Library empty.")

# === TAB 2: MANAGE LIBRARY ===
with tab_manager:
    st.subheader("🛠️ Prompt Manager")
    st.caption("Note: To edit a style, currently you have to Delete and Add New.")
    
    with st.expander("➕ Add New Style", expanded=False):
        with st.form("add_prompt"):
            new_name = st.text_input("Style Name")
            new_cat = st.text_input("Category")
            new_temp = st.text_area("Template (Use {var} for variables)", "A model wearing {color} ring...")
            new_vars = st.text_input("Variables (comma separated)", "color, size")
            new_url = st.text_input("Sample Image URL")
            
            if st.form_submit_button("Save Style"):
                new_entry = {
                    "id": f"p{len(st.session_state.prompt_library) + 1000}",
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
            st.write(f"**{p['name']}**")
            st.caption(f"{p['template'][:50]}...")
        with c3:
            if st.button("Delete", key=f"del_{idx}"):
                st.session_state.prompt_library.pop(idx)
                save_prompts(st.session_state.prompt_library)
                st.rerun()
