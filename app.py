"""Streamlit Cloud app for Rock / Paper / Scissors hand-gesture recognition."""

from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

from landmarks import draw_landmarks, extract_landmarks

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "gesture_model.pkl"
GESTURES = {"rock": "✊", "paper": "✋", "scissors": "✌️"}

st.set_page_config(page_title="HandPlay | Rock Paper Scissors", page_icon="✋", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner="Loading the gesture model…")
def load_model():
    return joblib.load(MODEL_PATH)


def to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def classify(model, image_bgr: np.ndarray):
    features = extract_landmarks(image_bgr)
    if features is None:
        return None, None, None
    probabilities = model.predict_proba([features])[0]
    return model.predict([features])[0], probabilities, model.classes_


def inject_styles():
    st.markdown("""<style>
    .stApp { background:radial-gradient(circle at top left,#e9edff,#fbfbff 42%,#f5f1ff); }
    .block-container { max-width:1160px; padding-top:2.5rem; padding-bottom:3rem; }
    .hero{text-align:center;margin:.5rem auto 2.5rem}.hero h1{font-size:clamp(2.4rem,6vw,4.8rem);margin:0;letter-spacing:-.06em;color:#17142d}.hero p{color:#625e7c;font-size:1.1rem;margin:.75rem auto 0;max-width:620px}.eyebrow{color:#7657e7;font-weight:750;letter-spacing:.12em;font-size:.75rem;text-transform:uppercase}.card{background:rgba(255,255,255,.77);border:1px solid rgba(102,79,190,.14);border-radius:24px;padding:1.35rem;box-shadow:0 12px 38px rgba(50,37,104,.08)}.prediction{background:linear-gradient(135deg,#5c43cb,#9b64e9);color:white;border-radius:24px;padding:2.1rem 1.5rem;text-align:center;min-height:220px;display:flex;flex-direction:column;justify-content:center}.prediction .emoji{font-size:4.5rem;line-height:1}.prediction h2{color:white;font-size:2.1rem;margin:.5rem 0 .25rem}div[data-testid="stFileUploader"] section{background:#fff;border-radius:14px}
    </style>""", unsafe_allow_html=True)


def main():
    inject_styles()
    st.markdown("""<div class="hero"><div class="eyebrow">Computer vision playground</div><h1>HandPlay</h1><p>Show a rock, paper, or scissors gesture. Our landmark-based model will identify it in seconds.</p></div>""", unsafe_allow_html=True)
    if not MODEL_PATH.exists():
        st.error("The trained model is missing. Add `gesture_model.pkl` beside `app.py` and redeploy.")
        st.stop()
    try:
        model = load_model()
    except Exception as error:
        st.error(f"The gesture model could not be loaded: {error}")
        st.stop()

    source = st.radio("Choose an input", ["Use camera", "Upload a photo"], horizontal=True, label_visibility="collapsed")
    submitted_image = None
    left, right = st.columns([1.25, .75], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if source == "Use camera":
            st.subheader("Take a snapshot")
            st.caption("Hold one hand clearly in the frame, then take a photo.")
            captured = st.camera_input("Camera", label_visibility="collapsed")
            if captured is not None:
                submitted_image = Image.open(captured)
        else:
            st.subheader("Upload a photo")
            st.caption("Use a JPG or PNG with one hand visible.")
            uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded is not None:
                submitted_image = Image.open(uploaded)
        if submitted_image is None:
            st.info("Your image preview will appear here after you choose or capture a photo.")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        if submitted_image is None:
            st.markdown('<div class="prediction"><div class="emoji">✋</div><h2>Ready when you are</h2><span>Capture or upload a gesture to see the result.</span></div>', unsafe_allow_html=True)
        else:
            image_bgr = to_bgr(submitted_image)
            prediction, probabilities, classes = classify(model, image_bgr)
            if prediction is None:
                st.warning("No hand found. Try brighter light, a plain background, and keep your full hand in view.")
            else:
                confidence = float(probabilities.max())
                st.markdown(f'<div class="prediction"><div class="emoji">{GESTURES.get(prediction, "✋")}</div><h2>{prediction.title()}</h2><span>{confidence:.0%} confidence</span></div>', unsafe_allow_html=True)
                st.markdown("#### Confidence by gesture")
                for label, probability in sorted(zip(classes, probabilities), key=lambda item: -item[1]):
                    st.progress(float(probability), text=f"{GESTURES.get(label, '')} {label.title()} — {probability:.1%}")
    if submitted_image is not None:
        st.markdown("#### Your hand landmarks")
        annotated = draw_landmarks(to_bgr(submitted_image))
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.divider()
    one, two, three = st.columns(3)
    one.markdown("### 1. Capture\nUse your camera or upload a clear photo.")
    two.markdown("### 2. Detect\nMediaPipe locates 21 points on your hand.")
    three.markdown("### 3. Play\nA trained classifier identifies your gesture.")
    with st.expander("About this project"):
        st.write("The model normalizes the 21 detected hand landmarks by wrist position and hand size. That makes its prediction less sensitive to where your hand is placed in the photo.")


if __name__ == "__main__":
    main()
