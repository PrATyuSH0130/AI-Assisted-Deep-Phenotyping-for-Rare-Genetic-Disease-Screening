
import json
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from huggingface_hub import hf_hub_download


# ==========================================
# CONFIGURATION
# ==========================================

HF_REPO = "Pratyush13/rare-disease-screening-models"

CNN_FILE = "FINAL_CNN_34CLASS_RESNET18.pth"
VIT_FILE = "FINAL_VIT_B16_34CLASS_ALL140.pth"
MAPPING_FILE = "FINAL_CNN_34CLASS_CLASS_MAPPING.json"

NUM_CLASSES = 34

DEVICE = torch.device("cpu")


# ==========================================
# STREAMLIT PAGE
# ==========================================

st.set_page_config(
    page_title="AI-Assisted Rare Disease Screening",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 AI-Assisted Deep Phenotyping")
st.subheader("Rare Genetic Disease Screening in India")

st.warning(
    "Research prototype only. Predictions are not a medical diagnosis "
    "and must not replace professional clinical assessment."
)


# ==========================================
# DOWNLOAD FILES FROM HUGGING FACE
# ==========================================

@st.cache_data
def download_file(filename):

    return hf_hub_download(
        repo_id=HF_REPO,
        filename=filename
    )


# ==========================================
# LOAD CLASS MAPPING
# ==========================================

@st.cache_data
def load_classes():

    mapping_path = download_file(MAPPING_FILE)

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    if isinstance(mapping, list):
        classes = mapping

    elif isinstance(mapping, dict):

        if "class_to_idx" in mapping:
            class_to_idx = mapping["class_to_idx"]

            classes = sorted(
                class_to_idx,
                key=lambda x: class_to_idx[x]
            )

        elif "classes" in mapping:
            classes = mapping["classes"]

        elif all(isinstance(v, int) for v in mapping.values()):
            classes = sorted(
                mapping,
                key=lambda x: mapping[x]
            )

        else:
            raise ValueError("Invalid class mapping format")

    else:
        raise ValueError("Unsupported class mapping format")

    if len(classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"but found {len(classes)}"
        )

    return [str(c) for c in classes]


# ==========================================
# LOAD MODEL WEIGHTS
# ==========================================

def get_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        for key in ["state_dict", "model_state_dict", "model"]:

            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format")

    cleaned = {}

    for key, value in checkpoint.items():

        if key.startswith("module."):
            key = key.replace("module.", "", 1)

        cleaned[key] = value

    return cleaned


def load_weights(model, filename):

    path = download_file(filename)

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

    state_dict = get_state_dict(checkpoint)

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


# ==========================================
# LOAD CNN MODEL
# ==========================================

@st.cache_resource
def load_cnn():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return load_weights(
        model,
        CNN_FILE
    )


# ==========================================
# LOAD VIT MODEL
# ==========================================

@st.cache_resource
def load_vit():

    model = models.vit_b_16(weights=None)

    # Checkpoint contains:
    # heads.head.1.weight
    # heads.head.1.bias

    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.0),
        nn.Linear(
            model.heads.head.in_features,
            NUM_CLASSES
        )
    )

    return load_weights(
        model,
        VIT_FILE
    )


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict(model, image):

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        top_probs, top_indices = torch.topk(
            probabilities,
            k=5
        )

    return top_probs.cpu().tolist(), top_indices.cpu().tolist()


# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_all_models():

    cnn = load_cnn()
    vit = load_vit()

    return cnn, vit


# ==========================================
# MAIN APPLICATION
# ==========================================

try:

    with st.spinner("Loading AI models..."):

        classes = load_classes()

        cnn, vit = load_all_models()

    st.success("Both CNN and ViT models loaded successfully.")

    st.write("**Number of disease classes:**", len(classes))

except Exception as e:

    st.error("Model loading failed.")

    st.exception(e)

    st.stop()


# ==========================================
# IMAGE UPLOAD
# ==========================================

st.divider()

st.header("Upload Patient Image")

uploaded_file = st.file_uploader(
    "Upload a facial photograph",
    type=["jpg", "jpeg", "png"]
)


# ==========================================
# RUN PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Patient Image",
            use_container_width=True
        )

    with col2:

        st.subheader("Prediction")

        if st.button("Analyze Image", type="primary"):

            try:

                with st.spinner("Analyzing image..."):

                    cnn_probs, cnn_indices = predict(cnn, image)

                    vit_probs, vit_indices = predict(vit, image)

                st.success("Analysis completed.")

                st.divider()

                # CNN RESULTS

                st.subheader("CNN — ResNet18")

                for rank, (prob, idx) in enumerate(
                    zip(cnn_probs, cnn_indices),
                    start=1
                ):

                    disease = classes[idx]

                    st.write(
                        f"{rank}. Disease ID: {disease} "
                        f"— Score: {prob * 100:.2f}%"
                    )

                    st.progress(float(prob))

                st.divider()

                # VIT RESULTS

                st.subheader("Vision Transformer — ViT-B/16")

                for rank, (prob, idx) in enumerate(
                    zip(vit_probs, vit_indices),
                    start=1
                ):

                    disease = classes[idx]

                    st.write(
                        f"{rank}. Disease ID: {disease} "
                        f"— Score: {prob * 100:.2f}%"
                    )

                    st.progress(float(prob))

                st.info(
                    "These scores represent model outputs, "
                    "not confirmed disease probabilities or diagnoses."
                )

            except Exception as e:

                st.error("Prediction failed.")

                st.exception(e)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI-Assisted Deep Phenotyping for Rare Genetic Disease Screening in India"
)

st.caption(
    "Academic research prototype | CNN + Vision Transformer"
)
