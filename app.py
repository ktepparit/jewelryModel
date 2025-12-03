import streamlit as st
import json
import os
import requests
import base64
from io import BytesIO
from PIL import Image

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Jewelry AI Studio")

# Default Prompts (ข้อมูลเริ่มต้น)
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

PROMPT_DB_FILE = "prompt_library.json"

# --- 2. BACKEND LOGIC ---
def get_prompts():
    if not os.path.exists(PROMPT_DB_FILE): return DEFAULT_PROMPTS
    try:
        with open(PROMPT_DB_FILE, "r") as f: return json.load(f)
    except: return DEFAULT_PROMPTS

def save_prompts(data):
    with open(PROMPT_DB_FILE, "w") as f: json.dump(data, f)

def img_to_base64(img):
    buf = BytesIO()
    if img.mode == 'RGBA': img = img.convert('RGB')
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def generate_image(api_key, image_list, prompt):
    # Model: Gemini 3 Pro Image Preview
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key={api_key}"
    
    # 1. เริ่มต้นด้วย Prompt text
    parts = [{"text": f"Instruction: {prompt} \nConstraint: Keep the jewelry products in the input images EXACTLY as they are. Analyze all images to understand the 3D structure. Generate a realistic model wearing it."}]
    
    # 2. วนลูปรูปภาพทั้งหมด ใส่ลงไปใน Payload (Multi-image support)
    for img in image_list:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_to_base64(img)
            }
        })

    try:
        # ยิง API
        res = requests.post(url, json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.3}}, headers={"Content-Type": "application/json"})
        
        if res.status_code != 200: return None, f"API Error: {res.text}"
        
        # แกะข้อมูลตอบกลับ
        content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
        
        # รองรับ key ทั้ง 2 แบบ (inline_data / inlineData)
        if "inline_data" in content: return base64.b64decode(content["inline_data"]["data"]), None
        if "inlineData" in content: return base64.b64decode(content["inlineData"]["data"]), None
        if "text" in content: return None, f"Model returned text: {content['text']}"
        
        return None, "Unknown response format."
    except Exception as e: return None, str(e)

# --- 3. FRONTEND UI ---
if "library" not in st.session_state: st.session_state.library = get_prompts()

with st.sidebar:
    st.title("⚙️ Config")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("API Key Loaded")
    except:
        api_key = st.text_input("Gemini API Key", type="password")

st.title("💎 Jewelry AI Studio (Multi-Upload)")

tab1, tab2 = st.tabs(["✨ Generate Image", "📚 Library Manager"])

# === TAB 1: GENERATE ===
with tab1:
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        st.subheader("1. Upload Images")
        # --- จุดสำคัญ: accept_multiple_files=True ---
        files = st.file_uploader("Drop images here (Max 4 recommended)", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
        
        images_to_send = []
        if files:
            st.write(f"Selected {len(files)} images")
            cols = st.columns(4)
            for i, f in enumerate(files):
                img = Image.open(f)
                images_to_send.append(img)
                cols[i % 4].image(img, use_column_width=True) # แสดง Preview เล็กๆ
        else:
            st.info("👆 Upload at least one image.")

    with c2:
        st.subheader("2. Settings")
        lib = st.session_state.library
        cats = list(set(p['category'] for p in lib))
        sel_cat = st.selectbox("Category", cats) if cats else None
        
        filtered = [p for p in lib if p['category'] == sel_cat]
        
        if filtered:
            sel_style = st.selectbox("Style", filtered, format_func=lambda x: x['name'])
            
            # Show Thumbnail
            if sel_style.get("sample_url"): st.image(sel_style["sample_url"], width=100)
            
            # Dynamic Variables
            vars_list = [v.strip() for v in sel_style['variables'].split(",") if v.strip()]
            user_vals = {}
            if vars_list:
                cols_var = st.columns(len(vars_list))
                for idx, v in enumerate(vars_list):
                    user_vals[v] = cols_var[idx].text_input(v, placeholder="e.g. Gold")
            
            # Create Prompt
            final_prompt = sel_style['template']
            for k, v in user_vals.items(): final_prompt = final_prompt.replace(f"{{{k}}}", v)
            
            # Editable Prompt
            st.write("✏️ **Edit Prompt:**")
            prompt_edit = st.text_area("Final instruction", value=final_prompt, height=100)
            
            if st.button("🚀 GENERATE", type="primary", use_container_width=True):
                if not api_key or not images_to_send:
                    st.error("Please check API Key and Images")
                else:
                    with st.spinner(f"Analyzing {len(images_to_send)} images..."):
                        img_data, err = generate_image(api_key, images_to_send, prompt_edit)
                        if img_data:
                            st.balloons()
                            st.image(img_data, caption="Generated Result", use_column_width=True)
                            st.download_button("Download Image", img_data, "jewelry_gen.jpg", "image/jpeg")
                        else:
                            st.error(err)
        else:
            st.warning("No styles found in library.")

# === TAB 2: LIBRARY ===
with tab2:
    st.subheader("Add New Style")
    with st.form("new_style"):
        c1, c2 = st.columns(2)
        n = c1.text_input("Name")
        c = c2.text_input("Category")
        t = st.text_area("Template (Use {var} for variables)")
        v = st.text_input("Variables (comma separated)", "color, size")
        u = st.text_input("Sample Image URL")
        if st.form_submit_button("Add Style"):
            st.session_state.library.append({
                "id": str(len(st.session_state.library)+100), 
                "name": n, "category": c, "template": t, "variables": v, "sample_url": u
            })
            save_prompts(st.session_state.library)
            st.success("Added!"); st.rerun()
            
    st.divider()
    st.subheader("Existing Styles")
    for i, p in enumerate(st.session_state.library):
        c1, c2, c3 = st.columns([1, 4, 1])
        if p.get("sample_url"): c1.image(p["sample_url"], width=50)
        c2.write(f"**{p['name']}** ({p['category']})")
        if c3.button("Delete", key=f"del_{i}"):
            st.session_state.library.pop(i)
            save_prompts(st.session_state.library)
            st.rerun()
