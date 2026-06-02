import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


APP_TITLE = "Plant Disease Detection"
CLASS_NAMES_PATH = Path("class_names.json")
LOCAL_MODEL_PATH = Path("plant_disease_model.h5")
CACHE_MODEL_PATH = Path("/tmp/plant_disease_model.h5")
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
                max-width: 1100px;
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


def get_secret_or_env(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.environ.get(key, default)


def download_model_if_needed(model_url):
    if LOCAL_MODEL_PATH.exists():
        return LOCAL_MODEL_PATH

    if CACHE_MODEL_PATH.exists():
        return CACHE_MODEL_PATH

    if not model_url:
        return None

    with st.spinner("Mengunduh model deteksi penyakit tanaman..."):
        urllib.request.urlretrieve(
            model_url,
            CACHE_MODEL_PATH,
        )

    return CACHE_MODEL_PATH


@st.cache_resource(show_spinner=False)
def load_trained_model(model_url):
    model_path = download_model_if_needed(model_url)

    if model_path is None or not model_path.exists():
        return None

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    return model


@st.cache_data
def load_class_names():
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
                Anda dapat mengunggah file gambar atau mengambil foto langsung dari kamera.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_notice():
    st.warning(
        "Model belum tersedia di aplikasi. Tambahkan `MODEL_URL` di Streamlit Secrets "
        "atau upload `plant_disease_model.h5` ke root repository."
    )

    with st.expander("Cara mengatur MODEL_URL di Streamlit Cloud"):
        st.markdown(
            """
            1. Upload file model `.h5` ke tempat penyimpanan publik, misalnya GitHub Release, Hugging Face, atau cloud storage.
            2. Salin direct download URL model.
            3. Di Streamlit Cloud, buka **Manage app → Settings → Secrets**.
            4. Isi secrets seperti ini:

            ```toml
            MODEL_URL = "https://direct-link-ke-model/plant_disease_model.h5"
            ```

            5. Simpan, lalu reboot aplikasi.
            """
        )


def render_prediction(best_result):
    plant = best_result["plant"]
    disease = best_result["disease"]
    confidence = best_result["confidence"]
    label = best_result["label"]

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if is_healthy(label):
        st.success("Tanaman terdeteksi sehat.")
    else:
        st.error("Tanaman terdeteksi memiliki indikasi penyakit.")

    metric_1, metric_2, metric_3 = st.columns(3)

    with metric_1:
        st.metric(
            "Tanaman",
            plant,
        )

    with metric_2:
        st.metric(
            "Diagnosis",
            disease,
        )

    with metric_3:
        st.metric(
            "Keyakinan",
            f"{confidence:.2f}%",
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_top_predictions(results):
    st.subheader("Top 5 Prediksi")

    for number, item in enumerate(results[:5], start=1):
        label = f"{number}. {item['plant']} — {item['disease']}"
        value = max(0.0, min(1.0, item["confidence"] / 100))

        st.write(f"**{label}**")
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

    class_names = load_class_names()
    model_url = get_secret_or_env("MODEL_URL")
    model = load_trained_model(model_url)

    left_col, right_col = st.columns(
        [1.05, 0.95],
        gap="large",
    )

    with left_col:
        st.subheader("Input Gambar")
        st.markdown(
            '<p class="small-muted">Gunakan gambar daun yang jelas, tidak blur, dan cukup cahaya.</p>',
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

        if model is None:
            render_model_notice()
        elif image is None:
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
    main()
