import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path("plant_disease_model.keras")
CLASS_NAMES_PATH = Path("class_names.json")
IMAGE_SIZE = (224, 224)
TOP_K = 5


st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
)


@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    return model


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"File class_names tidak ditemukan: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return class_names


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def parse_label(label):
    if "___" in label:
        plant, disease = label.split("___", 1)
    else:
        plant = "Unknown"
        disease = label

    plant = plant.replace("_", " ")
    disease = disease.replace("_", " ")
    disease = disease.replace("(", " (")
    disease = " ".join(disease.split())

    return plant, disease


def label_is_healthy(label):
    return "healthy" in label.lower()


def predict(model, image, class_names):
    image_array = preprocess_image(image)

    probabilities = model.predict(
        image_array,
        verbose=0,
    )[0]

    ranked_indices = np.argsort(probabilities)[::-1]

    results = []

    for index in ranked_indices:
        index = int(index)
        label = class_names[index]
        plant, disease = parse_label(label)
        confidence = float(probabilities[index]) * 100

        results.append(
            {
                "label": label,
                "plant": plant,
                "disease": disease,
                "confidence": confidence,
            }
        )

    return results


def render_main_result(result):
    st.subheader("Hasil Deteksi")

    if label_is_healthy(result["label"]):
        st.success("Tanaman terdeteksi sehat.")
    else:
        st.error("Tanaman terdeteksi memiliki indikasi penyakit.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Tanaman",
            value=result["plant"],
        )

    with col2:
        st.metric(
            label="Keyakinan Model",
            value=f"{result['confidence']:.2f}%",
        )

    st.markdown("**Diagnosis:**")
    st.write(result["disease"])


def render_top_results(results):
    st.subheader("Top 5 Prediksi")

    for number, item in enumerate(results[:TOP_K], start=1):
        title = f"{number}. {item['plant']} — {item['disease']}"
        progress_value = max(0.0, min(1.0, item["confidence"] / 100))

        st.markdown(f"**{title}**")
        st.progress(
            progress_value,
            text=f"{item['confidence']:.2f}%",
        )


def render_supported_classes(class_names):
    with st.expander("Daftar kelas yang didukung model"):
        grouped_classes = {}

        for label in class_names:
            plant, disease = parse_label(label)

            if plant not in grouped_classes:
                grouped_classes[plant] = []

            grouped_classes[plant].append(disease)

        for plant in sorted(grouped_classes.keys()):
            st.markdown(f"**{plant}**")

            for disease in sorted(grouped_classes[plant]):
                st.markdown(f"- {disease}")


st.title("🌿 Plant Disease Detection")
st.write(
    "Upload gambar daun tanaman untuk mendeteksi kemungkinan penyakit "
    "berdasarkan model TensorFlow/Keras."
)

st.info(
    "Gunakan gambar daun yang jelas, fokus, dan memiliki pencahayaan cukup. "
    "Hasil prediksi sangat dipengaruhi kualitas foto."
)

try:
    model = load_trained_model()
    class_names = load_class_names()

    uploaded_file = st.file_uploader(
        "Upload gambar daun",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Gambar yang di-upload",
            use_container_width=True,
        )

        if st.button("Deteksi Penyakit", type="primary"):
            results = predict(
                model=model,
                image=image,
                class_names=class_names,
            )

            render_main_result(results[0])
            render_top_results(results)
    else:
        st.warning("Silakan upload gambar daun terlebih dahulu.")

    render_supported_classes(class_names)

    st.caption(
        "Catatan: aplikasi ini adalah alat bantu prediksi awal, "
        "bukan pengganti diagnosis ahli pertanian."
    )

except Exception as error:
    st.error("Aplikasi gagal dijalankan.")
    st.exception(error)
