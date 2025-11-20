import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import requests
import zipfile
import os
from gtts import gTTS
import base64
from io import BytesIO
import uuid

# Load local NLLB model
#MODEL_PATH = r"C:\Users\valkontek 010\Downloads\nllb_model_zip"
MODEL_URL = "https://huggingface.co/mahigodike/translator_model/resolve/main/nllb_model_zip/nllb_model_zip.zip"
MODEL_ZIP_PATH = "nllb_model.zip"
MODEL_PATH = "nllb_model"
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

# -----------------------------
# STEP 1: DOWNLOAD MODEL ZIP
# -----------------------------
def download_model():
    if not os.path.exists(MODEL_ZIP_PATH):
        st.info("Downloading model... Please wait 1–2 minutes ⏳")
        response = requests.get(MODEL_URL)
        with open(MODEL_ZIP_PATH, "wb") as f:
            f.write(response.content)
        

# -----------------------------
# STEP 2: UNZIP THE MODEL
# -----------------------------
def extract_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Extracting model files... 🔍")
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_PATH)
        
@st.cache_resource
def load_model():
    download_model()
    extract_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

# Supported language codes from YOUR tokenizer
indian_lang_codes = {
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Marathi": "mar_Deva",
    "Telugu": "tel_Telu",
    "Tamil": "tam_Taml",
    "Gujarati": "guj_Gujr",
    "Urdu": "urd_Arab",
    "Kannada": "kan_Knda",
    "Odia": "ory_Orya",
    "Malayalam": "mal_Mlym",
    "Punjabi": "pan_Guru",
    "Assamese": "asm_Beng",
    "Maithili": "mai_Deva",
    "Santali (Bengali Script)": "sat_Beng",
    "Bhojpuri": "bho_Deva",
    "Kashmiri": "kas_Arab",
    "Nepali": "npi_Deva",
    "Sindhi": "snd_Arab",
    "Meitei (Manipuri Bengali Script)": "mni_Beng",
    "Khasi": "kha_Latn",   # From your need
}

st.title("Valkontek Embedded & IOT Services PVT.LTD")
st.title("Indian Languages Translator")

# UI Layout: Google Translate Style
col1, col2 = st.columns(2)

with col1:
    src_lang_name = st.selectbox("Source Language", indian_lang_codes.keys())
    src_lang = indian_lang_codes[src_lang_name]
    text = st.text_area("Enter Text", height=200)
with col2:
    tgt_lang_name = st.selectbox("Target Language", indian_lang_codes.keys())
    tgt_lang = indian_lang_codes[tgt_lang_name]



if st.button("Translate"):
    if text.strip() == "":
        st.error("Please enter some text to translate.")
    else:
        tokenizer.src_lang = src_lang

        inputs = tokenizer(text, return_tensors="pt")
        bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=bos_id,
            max_length=400
        )

        translated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # SAVE OUTPUT IN SESSION
        st.session_state.output_text = translated

with col2:
    st.text_area("Translation", st.session_state.output_text, height=200)

# ----------------------
# SPEAK BUTTON SEPARATE
# ----------------------
with col2:

    if st.button("🔊 Speak Translation"):
        # Empty text check
        if "output_text" not in st.session_state or st.session_state.output_text.strip() == "":
            st.warning("Translate something first.")
        else:

            # Extract language code
            lang_prefix = tgt_lang.split("_")[0]

            # TTS with fallback exception
            try:
                tts = gTTS(st.session_state.output_text, lang=lang_prefix)
            except Exception as e:
                tts = gTTS(st.session_state.output_text, lang="en")

            # Save audio
            tts.save("tts_output.mp3")

            # Convert to base64
            audio_bytes = open("tts_output.mp3", "rb").read()
            audio_base64 = base64.b64encode(audio_bytes).decode()

            # NEW random ID each click → browser treats as NEW audio → autoplay works everytime
            unique_id = str(uuid.uuid4())

            # Auto-play HTML (hidden player)
            audio_html = f"""
                <audio id="{unique_id}" autoplay style="display:none;">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                </audio>
            """

            # Load audio in browser
            st.markdown(audio_html, unsafe_allow_html=True)
