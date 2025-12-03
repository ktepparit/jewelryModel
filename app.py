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
    
    parts = [{"text": f"Instruction: {prompt} \nConstraint: Keep the jewelry in input images EXACTLY as is. Generate realistic
