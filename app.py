import json
import os
import shutil
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


APP_TITLE = "Plant Disease Detection"

CLASS_NAMES_PATH = Path("class_names.json")
IMAGE_SIZE = (224, 224)

LOCAL_MODEL_CANDIDATES = [
    Path("plant_disease_model.keras"),
    Path("plant_disease_model.h5"),
]

DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/"
    "adiorany3/PlantDesease/main/plant_disease_model.keras"
)

TEMP_DIR = Path("/tmp/plant_disease_app")
TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DOWNLOADED_MODEL_PATH = TEMP_DIR / "plant_disease_model_downloaded.keras"
TEMP_H5_PATH = TEMP_DIR / "plant_disease_model_loaded_as_h5.h5"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


class InvalidModelFileError(Exception):
    pass


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


def get_secret_or_env(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.environ.get(key, default)


def read_file_head(path, size=512):
    with open(path, "rb") as file:
        return file.read(size)


def detect_model_format(path):
    if not path.exists():
        return "missing"

    head = read_file_head(path)

    if len(head) == 0:
        return "empty"

    if head.startswith(b"PK"):
        return "keras_zip"

    if head.startswith(b"\x89HDF"):
        return "hdf5"

    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return "git_lfs_pointer"

    stripped = head.lstrip().lower()

    if stripped.startswith(b"<!doctype html") or stripped.startswith(b"<html"):
        return "html"

    if stripped.startswith(b"{") or stripped.startswith(b"["):
        return "json_or_text"

    return "unknown"


def make_keras_loadable_path(path):
    detected_format = detect_model_format(path)

    if detected_format == "keras_zip":
        return path

    if detected_format == "hdf5":
        shutil.copy2(
            path,
            TEMP_H5_PATH,
        )

        return TEMP_H5_PATH

    raise InvalidModelFileError(
        f"File model tidak valid: {path}. "
        f"Format terdeteksi: {detected_format}. "
        "File model harus berupa Keras `.keras` zip atau HDF5 `.h5`."
    )


def find_local_model_path():
    for candidate in LOCAL_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    return None


def download_model_from_url(model_url):
    if DOWNLOADED_MODEL_PATH.exists():
        detected_format = detect_model_format(DOWNLOADED_MODEL_PATH)

        if detected_format in ["keras_zip", "hdf5"]:
            return DOWNLOADED_MODEL_PATH

        DOWNLOADED_MODEL_PATH.unlink()

    try:
        with urllib.request.urlopen(model_url, timeout=120) as response:
            with open(DOWNLOADED_MODEL_PATH, "wb") as output_file:
                shutil.copyfileobj(
                    response,
                    output_file,
                )

    except HTTPError as error:
        raise InvalidModelFileError(
            f"Gagal mengunduh model dari MODEL_URL. HTTP error: {error.code}."
        ) from error

    except URLError as error:
        raise InvalidModelFileError(
            f"Gagal mengunduh model dari MODEL_URL. Detail: {error}."
        ) from error

    return DOWNLOADED_MODEL_PATH


def resolve_model_path():
    model_url = get_secret_or_env(
        "MODEL_URL",
        DEFAULT_MODEL_URL,
    )

    local_model_path = find_local_model_path()

    if local_model_path is not None:
        local_format = detect_model_format(local_model_path)

        if local_format in ["keras_zip", "hdf5"]:
            return make_keras_loadable_path(local_model_path)

        # Git LFS pointer, HTML, or corrupt file:
        # keep the repo lightweight, but try to fetch the true raw model at runtime.
        if model_url:
            downloaded_path = download_model_from_url(model_url)
            downloaded_format = detect_model_format(downloaded_path)

            if downloaded_format in ["keras_zip", "hdf5"]:
                return make_keras_loadable_path(downloaded_path)

            raise InvalidModelFileError(
                "File model lokal dan file hasil download sama-sama tidak valid. "
                f"Format lokal: {local_format}. "
                f"Format download: {downloaded_format}. "
                "Jika file lokal adalah Git LFS pointer, pastikan Git LFS quota/bandwidth masih tersedia, "
                "atau upload model ke GitHub Release/Hugging Face lalu set MODEL_URL di Streamlit Secrets."
            )

        raise InvalidModelFileError(
            f"File model lokal ditemukan tetapi tidak valid. Format: {local_format}."
        )

    if model_url:
        downloaded_path = download_model_from_url(model_url)
        downloaded_format = detect_model_format(downloaded_path)

        if downloaded_format in ["keras_zip", "hdf5"]:
            return make_keras_loadable_path(downloaded_path)

        raise InvalidModelFileError(
            f"File hasil download dari MODEL_URL tidak valid. Format: {downloaded_format}."
        )

    available_files = [
        str(path)
        for path in Path(".").glob("*")
    ]

    raise FileNotFoundError(
        "File model tidak ditemukan. "
        "Pastikan `plant_disease_model.keras` ada di root repository, "
        "atau isi MODEL_URL di Streamlit Secrets. "
        f"File yang tersedia: {available_files}"
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
            f"File yang tersedia: {available_files}"
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


def render_model_error(error):
    st.error("Model belum dapat dimuat.")
    st.exception(error)

    with st.expander("Cara memperbaiki file model"):
        st.markdown(
            """
            Periksa salah satu opsi berikut:

            1. Pastikan `plant_disease_model.keras` di GitHub adalah file model asli, bukan Git LFS pointer.
            2. Jika memakai Git LFS, pastikan bandwidth/quota Git LFS belum habis.
            3. Cara paling stabil: upload model ke GitHub Release atau Hugging Face, lalu isi Streamlit Secrets:

            ```toml
            MODEL_URL = "https://direct-download-url/plant_disease_model.keras"
            ```

            Aplikasi ini bisa membaca:
            - native Keras `.keras` zip
            - HDF5 `.h5`, termasuk file HDF5 yang namanya masih `.keras`
            """
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

    class_names = load_class_names()

    try:
        with st.spinner("Memuat model deteksi..."):
            model = load_trained_model()

    except Exception as error:
        render_model_error(error)
        render_supported_classes(class_names)
        return

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
    main()
