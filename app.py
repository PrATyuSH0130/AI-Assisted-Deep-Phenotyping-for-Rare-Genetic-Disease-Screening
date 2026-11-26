
import json
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from huggingface_hub import hf_hub_download


# ---------------- CONFIG ----------------

st.set_page_config(
    page_title="Rare Disease Screening",
    page_icon="🧬",
    layout="wide"
)

HF_REPO = "Pratyush13/rare-disease-screening-models"

CNN_FILE = "FINAL_CNN_34CLASS_RESNET18.pth"
VIT_FILE = "FINAL_VIT_B16_34CLASS_ALL140.pth"
MAPPING_FILE = "FINAL_CNN_34CLASS_CLASS_MAPPING.json"

NUM_CLASSES = 34

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------- IMAGE PREPROCESSING ----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ---------------- CLASS MAPPING ----------------

@st.cache_data
def load_classes():

    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=MAPPING_FILE
    )

    with open(path, "r") as f:
        classes = json.load(f)

    if isinstance(classes, dict):
        if "class_to_idx" in classes:
            mapping = classes["class_to_idx"]
            classes = [
                name for name, idx in
                sorted(mapping.items(), key=lambda x: x[1])
            ]
        elif "classes" in classes:
            classes = classes["classes"]
        else:
            raise ValueError("Invalid class mapping format")

    if len(classes) != NUM_CLASSES:
        raise ValueError("Expected exactly 34 classes")

    return classes


# ---------------- CHECKPOINT LOADER ----------------

def get_state_dict(checkpoint):

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint")

    for key in ["state_dict", "model_state_dict", "model"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break

    keys = list(checkpoint.keys())

    if keys and all(k.startswith("module.") for k in keys):
        checkpoint = {
            k.replace("module.", "", 1): v
            for k, v in checkpoint.items()
        }

    return checkpoint


def load_weights(model, filename):

    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        get_state_dict(checkpoint),
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


# ---------------- CNN MODEL ----------------

@st.cache_resource
def load_cnn():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return load_weights(model, CNN_FILE)


# ---------------- ViT MODEL ----------------

@st.cache_resource
def load_vit():

    model = models.vit_b_16(weights=None)

    model.heads.head = nn.Linear(
        model.heads.head.in_features,
        NUM_CLASSES
    )

    return load_weights(model, VIT_FILE)


# ---------------- PREDICTION ----------------

def predict(model, image, classes):

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():

        output = model(image_tensor)

        if output.shape[1] != NUM_CLASSES:
            raise ValueError("Model output is not 34 classes")

        probabilities = torch.softmax(output, dim=1)[0]

        values, indices = torch.topk(
            probabilities, 5
        )

    results = []

    for value, index in zip(
        values.cpu().tolist(),
        indices.cpu().tolist()
    ):

        results.append({
            "Disease Class": classes[index],
            "Confidence (%)": round(value * 100, 2)
        })

    return results


# ---------------- USER INTERFACE ----------------

st.title("🧬 AI-Assisted Deep Phenotyping")

st.subheader(
    "Rare Genetic Disease Screening in India"
)

st.write(
    "CNN and Vision Transformer based facial "
    "phenotype classification research prototype."
)

st.info(
    "Upload an image to obtain predictions from both trained models."
)

st.caption(f"Device: {DEVICE}")

uploaded_file = st.file_uploader(
    "Upload facial image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        width=350
    )

    try:

        with st.spinner("Loading trained models..."):

            classes = load_classes()
            cnn = load_cnn()
            vit = load_vit()

        with st.spinner("Generating predictions..."):

            cnn_results = predict(cnn, image, classes)
            vit_results = predict(vit, image, classes)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("CNN — ResNet18")

            st.dataframe(
                cnn_results,
                use_container_width=True,
                hide_index=True
            )

        with col2:

            st.subheader("ViT — B/16")

            st.dataframe(
                vit_results,
                use_container_width=True,
                hide_index=True
            )

        st.success("Inference completed successfully.")

    except Exception as e:

        st.error("Model loading or prediction failed.")
        st.exception(e)

        st.warning(
            "Check the checkpoint architecture, "
            "class mapping and preprocessing."
        )


st.divider()

st.warning(
    "Research prototype only. Results are experimental "
    "and are not a medical diagnosis."
)

st.caption(
    "BIT Mesra | CNN + Vision Transformer | Research Project"
)
