import streamlit as st
import json
import requests
import base64
from io import BytesIO
from PIL import Image

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Jewelry AI Studio")

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

# --- 3. HELPER FUNCTIONS ---
def img_to_base64(img):
    buf = BytesIO()
    if img.mode == 'RGBA': img = img.convert('RGB')
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

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

# --- 4. UI LOGIC ---
if "library" not in st.session_state:
    st.session_state.library = get_prompts()

# State สำหรับการ Edit
if "edit_target" not in st.session_state:
    st.session_state.edit_target = None

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
tab1, tab2 = st.tabs(["✨ Generate Image", "📚 Library Manager"])

# === TAB 1: GENERATE ===
with tab1:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("1. Upload")
        files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
        images_to_send = [Image.open(f) for f in files] if files else []
        if images_to_send:
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
            
            if st.button("🚀 GENERATE", type="primary", use_container_width=True):
                if not api_key or not images_to_send: st.error("Check Key & Images")
                else:
                    with st.spinner("Generating..."):
                        d, e = generate_image(api_key, images_to_send, prompt_edit)
                        if d: st.image(d); st.download_button("Download", d, "gen.jpg")
                        else: st.error(e)
        else: st.warning("Library empty.")

# === TAB 2: LIBRARY (UPDATED WITH EDIT) ===
with tab2:
    st.subheader("🛠️ Prompt Library")
    
    # ดูว่าตอนนี้กำลัง Edit ใครอยู่หรือเปล่า?
    target = st.session_state.edit_target
    form_title = f"✏️ Edit Style: {target['name']}" if target else "➕ Add New Style"
    
    # 1. ฟอร์ม (ใช้ร่วมกันทั้ง Add และ Edit)
    with st.form("style_form"):
        st.write(f"**{form_title}**")
        c1, c2 = st.columns(2)
        
        # ถ้ามี target (กำลัง Edit) ให้ดึงค่าเก่ามาใส่, ถ้าไม่มี (Add New) ให้เป็นค่าว่าง
        n = c1.text_input("Name", value=target['name'] if target else "")
        c = c2.text_input("Category", value=target['category'] if target else "")
        t = st.text_area("Template", value=target['template'] if target else "A model wearing {color} ring...")
        v = st.text_input("Variables", value=target['variables'] if target else "color, size")
        u = st.text_input("Sample Image URL", value=target['sample_url'] if target else "")
        
        # ปุ่มกดในฟอร์ม
        cols = st.columns([1, 4])
        submitted = cols[0].form_submit_button("💾 Save Style")
        
        # ถ้ากำลัง Edit จะมีปุ่ม Cancel ให้ด้วย
        if target:
            if cols[1].form_submit_button("❌ Cancel Edit"):
                st.session_state.edit_target = None
                st.rerun()

        if submitted:
            # เตรียมข้อมูลใหม่
            new_data = {
                "id": target['id'] if target else str(len(st.session_state.library) + 1000),
                "name": n, "category": c, "template": t, "variables": v, "sample_url": u
            }
            
            if target:
                # กรณี Edit: หาตัวเดิมแล้วทับค่าลงไป
                for idx, item in enumerate(st.session_state.library):
                    if item['id'] == target['id']:
                        st.session_state.library[idx] = new_data
                        break
                st.success("Updated Successfully!")
            else:
                # กรณี Add: เพิ่มต่อท้าย
                st.session_state.library.append(new_data)
                st.success("Added Successfully!")
            
            # Save ลง Database
            save_prompts(st.session_state.library)
            
            # Reset state และ Refresh
            st.session_state.edit_target = None
            st.rerun()

    st.divider()
    
    # 2. รายการ Style (เพิ่มปุ่ม Edit)
    for i, p in enumerate(st.session_state.library):
        c1, c2, c3, c4 = st.columns([1, 4, 1, 1])
        if p.get("sample_url"): c1.image(p["sample_url"], width=50)
        c2.write(f"**{p['name']}** ({p['category']})")
        
        # ปุ่ม Edit
        if c3.button("✏️ Edit", key=f"edit_{i}"):
            st.session_state.edit_target = p # ส่งข้อมูลตัวนี้ขึ้นไปที่ฟอร์ม
            st.rerun()
            
        # ปุ่ม Delete
        if c4.button("🗑️ Del", key=f"del_{i}"):
            st.session_state.library.pop(i)
            save_prompts(st.session_state.library)
            st.rerun()
