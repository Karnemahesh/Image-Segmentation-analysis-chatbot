import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multi-Image Chat with Gemini",
    layout="wide"
)

# ---------------- READ API KEY ----------------
try:
    api_key = st.secrets["general"]["GOOGLE_API_KEY"]
except KeyError:
    st.error("❌ GOOGLE_API_KEY not set in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# ---------------- SESSION STATE ----------------
if "images" not in st.session_state:
    st.session_state.images = []

if "active_image" not in st.session_state:
    st.session_state.active_image = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- SAFE GEMINI CALL ----------------
def safe_generate_content(prompt, image_bytes=None):
    try:
        if image_bytes:
            image_data = {
                "mime_type": "image/jpeg",
                "data": image_bytes
            }
            response = model.generate_content([prompt, image_data])
        else:
            response = model.generate_content(prompt)

        return response.text

    except ResourceExhausted:
        st.warning("⚠️ Gemini API quota exceeded — running in offline mode.")
        return f"[Offline Mode] {prompt[:100]}..."

# ---------------- IMAGE ANALYSIS ----------------
def analyze_image(image_bytes):
    return {
        "DESCRIPTION": safe_generate_content(
            "Describe this image in detail, include context and objects you see.",
            image_bytes
        ),
        "CAPTION": safe_generate_content(
            "Write a short catchy caption for this image.",
            image_bytes
        ),
        "TAGS": safe_generate_content(
            "Generate 5 short tags for this image.",
            image_bytes
        ),
        "STORY": safe_generate_content(
            "Write a short 3-sentence fictional story inspired by this image.",
            image_bytes
        ),
    }

# ---------------- SIDEBAR ----------------
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

            analysis = analyze_image(img_bytes)

            st.session_state.images.append({
                "name": file.name,
                "data": img_bytes,
                "analysis": analysis
            })

            st.session_state.active_image = len(st.session_state.images) - 1

    st.markdown("### 🖼 Uploaded Images")

    for idx, img in enumerate(st.session_state.images):
        if st.button(img["name"], key=f"img-{idx}"):
            st.session_state.active_image = idx

# ---------------- MAIN UI ----------------
st.title("💬 Multi-Image Conversational Chatbot with Gemini")

if st.session_state.active_image is not None:
    img_obj = st.session_state.images[st.session_state.active_image]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(
            io.BytesIO(img_obj["data"]),
            caption=img_obj["name"],
            width=300
        )

    with col2:
        st.subheader("Image Analysis")
        st.markdown(f"**Description:** {img_obj['analysis']['DESCRIPTION']}")
        st.markdown(f"**Caption:** {img_obj['analysis']['CAPTION']}")
        st.markdown(f"**Tags:** {img_obj['analysis']['TAGS']}")
        st.markdown(f"**Story:** {img_obj['analysis']['STORY']}")

# ---------------- CHAT ----------------
st.subheader("💬 Conversation")

for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {msg}")

if user_msg := st.chat_input("Type your message..."):

    st.session_state.chat_history.append(("You", user_msg))

    if st.session_state.active_image is not None:
        img_obj = st.session_state.images[st.session_state.active_image]
        bot_reply = safe_generate_content(user_msg, img_obj["data"])
    else:
        bot_reply = safe_generate_content(user_msg)

    st.session_state.chat_history.append(("Bot", bot_reply))

