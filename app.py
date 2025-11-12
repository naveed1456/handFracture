import io
import os
import numpy as np
from PIL import Image
import streamlit as st

os.environ["YOLO_VERBOSE"] = "False"

try:
    from ultralytics import YOLO
except Exception as e:
    st.error("Ultralytics not installed. Run: pip install ultralytics")
    raise e

# ----- Config -----
MODEL_PATH = "best.pt"       # keep your best.pt next to this file
FRACTURE_NAME = "fracture"   # adjust if your class name is different
CONF_THRESH = 0.25           # decision threshold

st.set_page_config(page_title="Hand Fracture: Yes/No", page_icon="🦴", layout="centered")
st.title("🦴 Hand Fracture Detector")

@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

def find_fracture_id(names_dict, target=FRACTURE_NAME):
    """
    Try exact match, otherwise fall back to substring match (e.g. 'hand_fracture').
    """
    # exact
    for cid, cname in names_dict.items():
        if cname.lower().strip() == target.lower().strip():
            return cid
    # contains
    for cid, cname in names_dict.items():
        if target.lower().strip() in cname.lower().strip():
            return cid
    return None

uploaded = st.file_uploader("Upload a hand X-ray / image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded:
    model = load_model()
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    with st.spinner("Analyzing..."):
        results = model.predict(source=np.array(img), conf=CONF_THRESH, verbose=False)
    if not results:
        st.error("No results returned.")
        st.stop()

    res = results[0]
    names = res.names  # {class_id: class_name}
    fracture_id = find_fracture_id(names, FRACTURE_NAME)

    fracture_found = False
    top_conf = None
    idxs = []

    if res.boxes is not None and len(res.boxes) > 0 and fracture_id is not None:
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        idxs = [i for i, c in enumerate(cls) if c == fracture_id and conf[i] >= CONF_THRESH]
        if idxs:
            fracture_found = True
            top_conf = float(np.max(conf[idxs]))

    st.subheader("Result")
    if fracture_found:
        st.success(f"✅ Fracture detected (confidence: {top_conf:.2f})")
    else:
        if fracture_id is None:
            st.warning(f"Class '{FRACTURE_NAME}' not found in model labels: {list(names.values())}")
        st.info("❌ No fracture detected.")

    # ---- Highlight fracture boxes (if any) ----
    # We'll draw boxes only for the fracture class on a copy of the image
    try:
        import cv2  # ensure opencv-python-headless is installed
        img_np = np.array(img).copy()  # RGB

        if fracture_found:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()

            for i in idxs:
                x1, y1, x2, y2 = xyxy[i].astype(int)
                label = f"{names.get(fracture_id, 'fracture')} {confs[i]:.2f}"

                # draw rectangle + label (convert to BGR for cv2 drawing then back to RGB)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)  # box
                # label background
                ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 255), -1)
                # label text
                cv2.putText(img_bgr, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            st.image(img_np, caption="Fracture highlighted", use_container_width=True)
        else:
            st.image(img, caption="Uploaded image", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not highlight boxes automatically ({e}). Showing original image.")
        st.image(img, caption="Uploaded image", use_container_width=True)

else:
    st.info("Upload an image to get a YES/NO answer.")
