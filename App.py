import streamlit as st
from huggingface_hub import InferenceClient
import io

st.set_page_config(page_title="Image Captioning (HF SDK)", layout="wide")

# -------- READ TOKEN --------
try:
    HF_API_KEY = st.secrets["huggingface"]["HF_API_KEY"]
except KeyError:
    st.error("HF_API_KEY not found in Streamlit Secrets.")
    st.stop()

# -------- CREATE CLIENT --------
client = InferenceClient(
    model="Salesforce/blip-image-captioning-base",
    token=HF_API_KEY
)

# -------- SESSION STATE --------
if "images" not in st.session_state:
    st.session_state.images = []

if "active_image" not in st.session_state:
    st.session_state.active_image = None

# -------- IMAGE DESCRIPTION --------
def describe_image(image_bytes):
    try:
        result = client.image_to_text(image_bytes)
        return result[0]["generated_text"]
    except Exception as e:
        return f"Error: {str(e)}"

# -------- SIDEBAR --------
with st.sidebar:
    st.header("Upload Images")

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            img_bytes = file.read()

            with st.spinner("Generating description..."):
                description = describe_image(img_bytes)

            st.session_state.images.append({
                "name": file.name,
                "data": img_bytes,
                "description": description
            })

            st.session_state.active_image = len(st.session_state.images) - 1

    st.markdown("### Uploaded Images")

    for idx, img in enumerate(st.session_state.images):
        if st.button(img["name"], key=f"img-{idx}"):
            st.session_state.active_image = idx

# -------- MAIN --------
st.title("Free Image Captioning App")

if st.session_state.active_image is not None:
    img_obj = st.session_state.images[st.session_state.active_image]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(io.BytesIO(img_obj["data"]), width=300)

    with col2:
        st.subheader("Generated Description")
        st.write(img_obj["description"])
