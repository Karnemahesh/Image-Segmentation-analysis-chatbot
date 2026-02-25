import streamlit as st
import requests
import io

st.set_page_config(page_title="Multi-Image Chat (HuggingFace)", layout="wide")

# ---------- READ HF API KEY ----------
try:
    HF_API_KEY = st.secrets["huggingface"]["HF_API_KEY"]
except KeyError:
    st.error("HF_API_KEY not found in Streamlit Secrets.")
    st.stop()

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# ---------- SESSION STATE ----------
if "images" not in st.session_state:
    st.session_state.images = []

if "active_image" not in st.session_state:
    st.session_state.active_image = None

# ---------- IMAGE DESCRIPTION ----------
def describe_image(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    result = response.json()

    if isinstance(result, list):
        return result[0]["generated_text"]
    else:
        return "Error generating description."

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("📂 Upload Images")

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            img_bytes = file.read()
            description = describe_image(img_bytes)

            st.session_state.images.append({
                "name": file.name,
                "data": img_bytes,
                "description": description
            })

            st.session_state.active_image = len(st.session_state.images) - 1

    st.markdown("### 🖼 Uploaded Images")

    for idx, img in enumerate(st.session_state.images):
        if st.button(img["name"], key=f"img-{idx}"):
            st.session_state.active_image = idx

# ---------- MAIN ----------
st.title("🖼 Image Description App (Free Hugging Face API)")

if st.session_state.active_image is not None:
    img_obj = st.session_state.images[st.session_state.active_image]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(io.BytesIO(img_obj["data"]), width=300)

    with col2:
        st.subheader("Generated Description")
        st.write(img_obj["description"])
