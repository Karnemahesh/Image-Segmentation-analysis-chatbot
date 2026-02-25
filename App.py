import streamlit as st
import requests
import io

st.set_page_config(
    page_title="Multi-Image Chat (HuggingFace)",
    layout="wide"
)

# ---------- READ HF API KEY ----------
try:
    HF_API_KEY = st.secrets["huggingface"]["HF_API_KEY"]
except KeyError:
    st.error("HF_API_KEY not found in Streamlit Secrets.")
    st.stop()

API_URL = "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/octet-stream"
}

# ---------- SESSION STATE ----------
if "images" not in st.session_state:
    st.session_state.images = []

if "active_image" not in st.session_state:
    st.session_state.active_image = None

# ---------- SAFE IMAGE DESCRIPTION ----------
def describe_image(image_bytes):
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            data=image_bytes,
            timeout=30
        )

        # If model is loading
        if response.status_code == 503:
            return "⏳ Model is loading... Please try again in a few seconds."

        # If rate limited or error
        if response.status_code != 200:
            return f"❌ API Error {response.status_code}: {response.text}"

        try:
            result = response.json()
        except Exception:
            return "❌ Invalid response received from API."

        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No description generated.")

        if isinstance(result, dict) and "error" in result:
            return f"⚠️ {result['error']}"

        return "⚠️ Unexpected API response."

    except requests.exceptions.Timeout:
        return "⏰ Request timed out. Please try again."

    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {str(e)}"


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

            # Avoid duplicate uploads
            if not any(img["name"] == file.name for img in st.session_state.images):

                with st.spinner("Generating description..."):
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

