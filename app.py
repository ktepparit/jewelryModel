import streamlit as st
import json
import os
import requests
import base64
from io import BytesIO
from PIL import Image

# --- CONFIG ---
st.set_page_config(layout="wide", page_title="Jewelry AI Studio")
PROMPT_DB_FILE = "prompt_library.json"

# --- DEFAULT DATA ---
DEFAULT_PROMPTS = [
    {
        "id": "p1", "name": "Luxury Hand (Ring)", "category": "Ring",
        "template": "A realistic close-up of a female hand model wearing a ring with {face_size} face size, soft studio lighting.",
        "variables": "face_size"
    },
    {
        "id": "p2", "name": "Streetwear Necklace", "category": "Necklace",
        "template": "A fashion portrait of a model wearing a {length} necklace, streetwear outfit, urban background.",
        "variables": "length"
    }
]

# --- FUNCTIONS ---
def get_prompts():
    if not os.path.exists(PROMPT_DB_FILE):
        return DEFAULT_PROMPTS
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

def generate_image(api_key, images, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent?key={api_key}"
    
    parts = [{"text": f"Instruction: {prompt} \nConstraint: Keep the jewelry in input images EXACTLY as is. Generate realistic model wearing it."}]
    for img in images:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}})

    try:
        res = requests.post(url, json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.3}}, headers={"Content-Type": "application/json"})
        if res.status_code != 200: return None, res.text
        
        content = res.json().get("candidates", [])[0].get("content", {}).get("parts", [])[0]
        # รองรับ Key ทั้ง 2 แบบ
        if "inline_data" in content: return base64.b64decode(content["inline_data"]["data"]), None
        if "inlineData" in content: return base64.b64decode(content["inlineData"]["data"]), None
        return None, "No image returned."
    except Exception as e: return None, str(e)

# --- UI ---
if "library" not in st.session_state: st.session_state.library = get_prompts()

with st.sidebar:
    st.title("⚙️ Config")
    api_key = st.text_input("Gemini API Key", type="password")

st.title("💎 Jewelry AI Studio")
tab1, tab2 = st.tabs(["✨ Generate", "📚 Library"])

with tab1:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        # Standard Uploader (รับหลายไฟล์ได้)
        files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png"])
        imgs = [Image.open(f) for f in files] if files else []
        if imgs:
            st.image(imgs, width=120, caption=[f"Img {i+1}" for i in range(len(imgs))])

    with c2:
        lib = st.session_state.library
        cats = list(set(p['category'] for p in lib))
        sel_cat = st.selectbox("Category", cats) if cats else None
        
        filtered = [p for p in lib if p['category'] == sel_cat]
        sel_style = st.selectbox("Style", filtered, format_func=lambda x: x['name']) if filtered else None

        final_prompt = ""
        if sel_style:
            vars = [v.strip() for v in sel_style['variables'].split(",") if v.strip()]
            user_vals = {v: st.text_input(v) for v in vars}
            final_prompt = sel_style['template']
            for k, v in user_vals.items(): final_prompt = final_prompt.replace(f"{{{k}}}", v)
        
        prompt_edit = st.text_area("Final Prompt", value=final_prompt, height=100)
        
        if st.button("🚀 Generate", type="primary"):
            if not api_key or not imgs: st.error("Need API Key & Images")
            else:
                with st.spinner("Generating..."):
                    img_data, err = generate_image(api_key, imgs, prompt_edit)
                    if img_data: st.image(img_data); st.download_button("Download", img_data, "gen.jpg")
                    else: st.error(err)

with tab2:
    with st.form("new"):
        n = st.text_input("Name")
        c = st.text_input("Category")
        t = st.text_area("Template")
        v = st.text_input("Vars (comma sep)")
        if st.form_submit_button("Add"):
            st.session_state.library.append({"id": str(len(st.session_state.library)), "name": n, "category": c, "template": t, "variables": v})
            save_prompts(st.session_state.library)
            st.rerun()
            
    for i, p in enumerate(st.session_state.library):
        c1, c2 = st.columns([4, 1])
        c1.write(f"**{p['name']}** ({p['category']})")
        if c2.button("Del", key=i):
            st.session_state.library.pop(i)
            save_prompts(st.session_state.library)
            st.rerun()
