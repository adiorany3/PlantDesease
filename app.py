import json
import shutil
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


APP_TITLE = "Plant Disease Detection"

MODEL_PATH = Path("plant_disease_model.keras")
MODEL_H5_FALLBACK_PATH = Path("plant_disease_model.h5")
TEMP_H5_PATH = Path("/tmp/plant_disease_model_from_keras_name.h5")

CLASS_NAMES_PATH = Path("class_names.json")
IMAGE_SIZE = (224, 224)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_custom_css():
    st.markdown(
        """
        <style>
            #MainMenu {
                visibility: hidden;
            }

            footer {
                visibility: hidden;
            }

            header {
                visibility: hidden;
            }

            [data-testid="stDecoration"] {
                display: none;
            }

            [data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }

            [data-testid="stDeployButton"] {
                display: none;
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1120px;
            }

            .hero-card {
                padding: 1.6rem 1.8rem;
                border-radius: 1.4rem;
                background: linear-gradient(135deg, #effaf1 0%, #ffffff 55%, #e8f5e9 100%);
                border: 1px solid #d8eadb;
                box-shadow: 0 8px 26px rgba(46, 125, 50, 0.08);
                margin-bottom: 1.2rem;
            }

            .hero-title {
                font-size: 2.25rem;
                font-weight: 800;
                color: #1b5e20;
                margin-bottom: 0.25rem;
            }

            .hero-subtitle {
                font-size: 1.02rem;
                color: #3f4f42;
                line-height: 1.7;
                margin-bottom: 0;
            }

            .result-card {
                padding: 1.2rem;
                border-radius: 1.2rem;
                border: 1px solid #dceee0;
                background-color: #ffffff;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.045);
            }

            .small-muted {
                color: #637067;
                font-size: 0.9rem;
            }

            .stButton > button {
                border-radius: 999px;
                font-weight: 700;
                padding-left: 1.3rem;
                padding-right: 1.3rem;
            }

            .stFileUploader, [data-testid="stCameraInput"] {
                border-radius: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def file_signature(path, size=8):
    with open(path, "rb") as file:
        return file.read(size)


def resolve_model_path():
    if MODEL_PATH.exists():
        signature = file_signature(MODEL_PATH)

        # Native Keras .keras format is a ZIP archive and starts with PK.
        if signature.startswith(b"PK"):
            return MODEL_PATH

        # Some training/export pipelines save HDF5 content but give it a .keras name.
        # Keras may reject that because the extension says .keras. Copy it to .h5 first.
        if signature.startswith(b"\x89HDF"):
            shutil.copy2(
                MODEL_PATH,
                TEMP_H5_PATH,
            )
            return TEMP_H5_PATH

        return MODEL_PATH

    if MODEL_H5_FALLBACK_PATH.exists():
        return MODEL_H5_FALLBACK_PATH

    available_files = [
        str(path)
        for path in Path(".").glob("*")
    ]

    raise FileNotFoundError(
        "File model tidak ditemukan. "
        "Pastikan file model sudah ada di root repository dengan nama "
        "`plant_disease_model.keras` atau `plant_disease_model.h5`. "
        f"File yang tersedia saat ini: {available_files}"
    )


@st.cache_resource(show_spinner=False)
def load_trained_model():
    model_path = resolve_model_path()

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    return model


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        available_files = [
            str(path)
            for path in Path(".").glob("*")
        ]

        raise FileNotFoundError(
            "File class_names.json tidak ditemukan. "
            f"File yang tersedia saat ini: {available_files}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    return names


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def split_label(label):
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


def is_healthy(label):
    return "healthy" in label.lower()


def predict(model, image, class_names):
    input_image = preprocess_image(image)

    predictions = model.predict(
        input_image,
        verbose=0,
    )[0]

    top_indices = np.argsort(predictions)[::-1]

    results = []

    for index in top_indices:
        label = class_names[int(index)]
        confidence = float(predictions[int(index)]) * 100
        plant, disease = split_label(label)

        results.append(
            {
                "label": label,
                "plant": plant,
                "disease": disease,
                "confidence": confidence,
            }
        )

    return results


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🌿 Plant Disease Detection</div>
            <p class="hero-subtitle">
                Deteksi awal penyakit tanaman dari gambar daun. 
                Unggah gambar dari file atau ambil foto langsung melalui kamera.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction(best_result):
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if is_healthy(best_result["label"]):
        st.success("Tanaman terdeteksi sehat.")
    else:
        st.error("Tanaman terdeteksi memiliki indikasi penyakit.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Tanaman",
            best_result["plant"],
        )

    with col2:
        st.metric(
            "Diagnosis",
            best_result["disease"],
        )

    with col3:
        st.metric(
            "Keyakinan",
            f"{best_result['confidence']:.2f}%",
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_top_predictions(results):
    st.subheader("Top 5 Prediksi")

    for number, item in enumerate(results[:5], start=1):
        text = f"{number}. {item['plant']} — {item['disease']}"
        value = max(
            0.0,
            min(
                1.0,
                item["confidence"] / 100,
            ),
        )

        st.write(f"**{text}**")
        st.progress(
            value,
            text=f"{item['confidence']:.2f}%",
        )


def render_supported_classes(class_names):
    with st.expander("Daftar tanaman dan penyakit yang didukung"):
        grouped = {}

        for label in class_names:
            plant, disease = split_label(label)

            if plant not in grouped:
                grouped[plant] = []

            grouped[plant].append(disease)

        columns = st.columns(2)

        for index, plant in enumerate(sorted(grouped.keys())):
            with columns[index % 2]:
                st.markdown(f"**{plant}**")

                for disease in sorted(grouped[plant]):
                    st.markdown(f"- {disease}")


def get_input_image():
    upload_tab, camera_tab = st.tabs(
        [
            "Upload Gambar",
            "Ambil Foto Langsung",
        ]
    )

    image = None

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Pilih file gambar daun",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    with camera_tab:
        camera_file = st.camera_input(
            "Ambil foto daun dari kamera"
        )

        if camera_file is not None:
            image = Image.open(camera_file)

    return image


def main():
    inject_custom_css()
    render_hero()

    with st.spinner("Memuat model dan daftar kelas..."):
        model = load_trained_model()
        class_names = load_class_names()

    left_col, right_col = st.columns(
        [1.05, 0.95],
        gap="large",
    )

    with left_col:
        st.subheader("Input Gambar")
        st.markdown(
            '<p class="small-muted">Gunakan foto daun yang jelas, tidak blur, dan pencahayaan cukup.</p>',
            unsafe_allow_html=True,
        )

        image = get_input_image()

        if image is not None:
            st.image(
                image,
                caption="Gambar yang akan dianalisis",
                use_container_width=True,
            )

    with right_col:
        st.subheader("Analisis")

        if image is None:
            st.info("Upload gambar atau ambil foto daun terlebih dahulu.")
        else:
            if st.button(
                "Deteksi Penyakit",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Menganalisis gambar daun..."):
                    results = predict(
                        model=model,
                        image=image,
                        class_names=class_names,
                    )

                render_prediction(results[0])
                render_top_predictions(results)

    render_supported_classes(class_names)

    st.caption(
        "Catatan: hasil prediksi adalah bantuan awal berbasis model dan bukan pengganti pemeriksaan ahli pertanian."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        st.error("Aplikasi gagal dijalankan.")
        st.exception(error)
