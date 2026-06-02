import json
import os
import shutil
import urllib.request
from collections import defaultdict
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

KAGGLE_TRAINING_URL = (
    "https://www.kaggle.com/code/adioranye/"
    "plant-disease-detection-by-adioranye"
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
            :root {
                color-scheme: light dark;

                --app-bg: #f7fbf7;
                --app-surface: #ffffff;
                --app-surface-2: #f1f8f2;
                --app-text: #17231a;
                --app-text-soft: #405047;
                --app-muted: #607064;
                --app-border: #d7eadb;
                --app-border-strong: #b9d9bf;
                --app-primary: #1b5e20;
                --app-primary-2: #2e7d32;
                --app-primary-soft: #e8f5e9;
                --app-shadow: 0 10px 28px rgba(31, 93, 37, 0.10);
                --app-danger-bg: #fff3f2;
                --app-danger-border: #ffc9c4;
                --app-success-bg: #eef9f0;
                --app-success-border: #bfe7c7;
                --app-warning-bg: #fff8e5;
                --app-warning-border: #ead083;
            }

            @media (prefers-color-scheme: dark) {
                :root {
                    --app-bg: #0e1510;
                    --app-surface: #151f18;
                    --app-surface-2: #19261d;
                    --app-text: #edf7ef;
                    --app-text-soft: #c9d9cd;
                    --app-muted: #a7b9ab;
                    --app-border: #2e4734;
                    --app-border-strong: #3e6047;
                    --app-primary: #8ee59a;
                    --app-primary-2: #6fd47d;
                    --app-primary-soft: #16331c;
                    --app-shadow: 0 10px 28px rgba(0, 0, 0, 0.26);
                    --app-danger-bg: #3a1717;
                    --app-danger-border: #7a3333;
                    --app-success-bg: #12361a;
                    --app-success-border: #2d7540;
                    --app-warning-bg: #3c2f10;
                    --app-warning-border: #82682b;
                }
            }

            html, body, [data-testid="stAppViewContainer"] {
                background: var(--app-bg);
                color: var(--app-text);
            }

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
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
                max-width: 1180px;
            }

            .hero-card,
            .section-card,
            .result-card,
            .insight-card,
            .custom-footer {
                background: var(--app-surface);
                color: var(--app-text);
                border: 1px solid var(--app-border);
                box-shadow: var(--app-shadow);
            }

            .hero-card {
                padding: clamp(1rem, 3vw, 1.8rem);
                border-radius: clamp(1rem, 2vw, 1.5rem);
                background:
                    radial-gradient(circle at top left, var(--app-primary-soft), transparent 36%),
                    linear-gradient(135deg, var(--app-surface-2) 0%, var(--app-surface) 62%);
                margin-bottom: 1.1rem;
            }

            .hero-title {
                font-size: clamp(1.65rem, 4vw, 2.45rem);
                font-weight: 850;
                color: var(--app-primary);
                margin-bottom: 0.35rem;
                line-height: 1.15;
                letter-spacing: -0.02em;
            }

            .hero-subtitle {
                font-size: clamp(0.92rem, 2.2vw, 1.04rem);
                color: var(--app-text-soft);
                line-height: 1.7;
                margin-bottom: 0;
                max-width: 880px;
            }

            .section-card,
            .result-card,
            .insight-card {
                padding: clamp(0.9rem, 2.2vw, 1.2rem);
                border-radius: 1.2rem;
                margin-bottom: 1rem;
            }

            .insight-card {
                background:
                    linear-gradient(135deg, var(--app-surface) 0%, var(--app-surface-2) 100%);
            }

            .small-muted,
            .insight-text {
                color: var(--app-muted);
                font-size: clamp(0.84rem, 1.9vw, 0.93rem);
                line-height: 1.65;
            }

            .insight-title {
                color: var(--app-primary);
                font-weight: 800;
                margin-bottom: 0.35rem;
                font-size: 1rem;
            }

            .insight-list {
                margin-top: 0.2rem;
                margin-bottom: 0;
                color: var(--app-text-soft);
                line-height: 1.65;
            }

            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 0.8rem;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.45rem 0.72rem;
                border-radius: 999px;
                background: var(--app-surface-2);
                color: var(--app-text);
                border: 1px solid var(--app-border);
                font-size: 0.88rem;
                font-weight: 700;
            }

            .stButton > button {
                border-radius: 999px;
                font-weight: 750;
                min-height: 2.75rem;
            }

            .stFileUploader, [data-testid="stCameraInput"] {
                border-radius: 1rem;
            }

            [data-testid="stImage"] img {
                border-radius: 1rem;
                border: 1px solid var(--app-border);
            }

            div[data-testid="stMetric"] {
                background: var(--app-surface-2);
                border: 1px solid var(--app-border);
                padding: 0.85rem 0.9rem;
                border-radius: 1rem;
                color: var(--app-text);
            }

            div[data-testid="stMetric"] label {
                color: var(--app-muted) !important;
            }

            div[data-testid="stMetricValue"] {
                color: var(--app-text) !important;
                font-size: clamp(1rem, 2vw, 1.25rem);
            }

            .custom-footer {
                margin-top: 2rem;
                padding: 1rem 1.2rem;
                border-radius: 1rem;
                text-align: center;
                color: var(--app-text-soft);
                font-size: clamp(0.78rem, 1.8vw, 0.92rem);
                line-height: 1.7;
            }

            .custom-footer a {
                color: var(--app-primary);
                font-weight: 800;
                text-decoration: none;
            }

            .custom-footer a:hover {
                text-decoration: underline;
            }

            .stAlert {
                color: var(--app-text);
            }

            [data-testid="stExpander"] {
                border-color: var(--app-border) !important;
                background: var(--app-surface) !important;
                color: var(--app-text) !important;
            }

            @media screen and (max-width: 900px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                    padding-top: 1rem;
                }

                .hero-card {
                    margin-bottom: 0.8rem;
                }

                .section-card,
                .result-card,
                .insight-card {
                    border-radius: 1rem;
                }

                .badge-row {
                    gap: 0.45rem;
                }
            }

            @media screen and (max-width: 640px) {
                .block-container {
                    padding-left: 0.75rem;
                    padding-right: 0.75rem;
                    padding-bottom: 1.5rem;
                }

                .hero-card {
                    padding: 1rem;
                    border-radius: 1rem;
                }

                .hero-title {
                    font-size: 1.55rem;
                }

                .hero-subtitle {
                    font-size: 0.9rem;
                    line-height: 1.55;
                }

                .stTabs [data-baseweb="tab-list"] {
                    gap: 0.35rem;
                }

                .stTabs [data-baseweb="tab"] {
                    padding-left: 0.55rem;
                    padding-right: 0.55rem;
                    font-size: 0.85rem;
                }

                div[data-testid="stMetric"] {
                    padding: 0.72rem;
                }

                .custom-footer {
                    padding-left: 0.75rem;
                    padding-right: 0.75rem;
                }

                .badge {
                    font-size: 0.8rem;
                    padding: 0.38rem 0.58rem;
                }
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


def summarize_classes(class_names):
    grouped = defaultdict(list)

    for label in class_names:
        plant, disease = split_label(label)
        grouped[plant].append(disease)

    healthy_classes = [
        label
        for label in class_names
        if is_healthy(label)
    ]

    disease_classes = [
        label
        for label in class_names
        if not is_healthy(label)
    ]

    top_plants = sorted(
        grouped.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )[:5]

    return grouped, healthy_classes, disease_classes, top_plants


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


def render_dataset_insights(class_names):
    grouped, healthy_classes, disease_classes, top_plants = summarize_classes(class_names)

    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.subheader("Insight Model")

    badge_html = f"""
    <div class="badge-row">
        <span class="badge">🌱 {len(grouped)} tanaman</span>
        <span class="badge">🧪 {len(class_names)} kelas</span>
        <span class="badge">🩺 {len(disease_classes)} kelas penyakit</span>
        <span class="badge">✅ {len(healthy_classes)} kelas sehat</span>
    </div>
    """

    st.markdown(
        badge_html,
        unsafe_allow_html=True,
    )

    top_plant_text = ", ".join(
        [
            f"{plant} ({len(diseases)} kelas)"
            for plant, diseases in top_plants
        ]
    )

    st.markdown(
        f"""
        <p class="insight-text">
            Model ini paling banyak memiliki variasi kelas pada: <strong>{top_plant_text}</strong>.
            Semakin mirip tanaman pada gambar dengan kelas yang tersedia, semakin relevan hasil prediksi.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_insights(best_result, second_result=None):
    confidence = best_result["confidence"]
    label = best_result["label"]

    if confidence >= 85:
        confidence_text = (
            "Confidence tinggi. Hasil ini cukup kuat sebagai deteksi awal, "
            "dengan catatan foto sesuai pola dataset pelatihan."
        )
    elif confidence >= 60:
        confidence_text = (
            "Confidence sedang. Sebaiknya cek top prediksi lain dan ulangi foto "
            "dengan pencahayaan lebih baik."
        )
    else:
        confidence_text = (
            "Confidence rendah. Model belum yakin; gunakan foto daun yang lebih dekat, "
            "lebih tajam, dan latar belakang lebih sederhana."
        )

    if is_healthy(label):
        status_text = (
            "Prediksi utama adalah kelas sehat. Tetap periksa gejala visual seperti bercak, "
            "perubahan warna, atau daun mengering jika kondisi lapangan meragukan."
        )
    else:
        status_text = (
            "Prediksi utama mengarah ke penyakit. Gunakan hasil ini sebagai prioritas awal "
            "untuk pemeriksaan lebih lanjut, bukan diagnosis final."
        )

    gap_text = ""

    if second_result is not None:
        gap = confidence - second_result["confidence"]

        if gap < 10:
            gap_text = (
                f"Selisih dengan prediksi kedua hanya {gap:.2f}%. "
                "Artinya beberapa kelas terlihat mirip bagi model."
            )
        else:
            gap_text = (
                f"Selisih dengan prediksi kedua {gap:.2f}%, sehingga prediksi utama "
                "lebih dominan dibanding alternatif berikutnya."
            )

    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.subheader("Insight Hasil")

    st.markdown(
        f"""
        <ul class="insight-list">
            <li>{confidence_text}</li>
            <li>{status_text}</li>
            <li>{gap_text}</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction(best_result):
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if is_healthy(best_result["label"]):
        st.success("Tanaman terdeteksi sehat.")
    else:
        st.error("Tanaman terdeteksi memiliki indikasi penyakit.")

    metric_cols = st.columns(3)

    metric_cols[0].metric(
        "Tanaman",
        best_result["plant"],
    )

    metric_cols[1].metric(
        "Diagnosis",
        best_result["disease"],
    )

    metric_cols[2].metric(
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
        grouped, _, _, _ = summarize_classes(class_names)
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


def render_footer():
    st.markdown(
        f"""
        <div class="custom-footer">
            Developed by <strong>Galuh Adi Insani</strong>, training with
            <a href="{KAGGLE_TRAINING_URL}" target="_blank" rel="noopener noreferrer">
                Kaggle
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_custom_css()
    render_hero()

    class_names = load_class_names()
    render_dataset_insights(class_names)

    try:
        with st.spinner("Memuat model deteksi..."):
            model = load_trained_model()

    except Exception as error:
        render_model_error(error)
        render_supported_classes(class_names)
        render_footer()
        return

    left_col, right_col = st.columns(
        [1.05, 0.95],
        gap="large",
    )

    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Input Gambar")
        st.markdown(
            '<p class="small-muted">Gunakan foto daun yang jelas, tidak blur, pencahayaan cukup, dan fokus pada area daun.</p>',
            unsafe_allow_html=True,
        )

        image = get_input_image()

        if image is not None:
            st.image(
                image,
                caption="Gambar yang akan dianalisis",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
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

                second_result = results[1] if len(results) > 1 else None
                render_prediction_insights(results[0], second_result)

                render_top_predictions(results)

        st.markdown("</div>", unsafe_allow_html=True)

    render_supported_classes(class_names)
    st.caption(
        "Catatan: hasil prediksi adalah bantuan awal berbasis model dan bukan pengganti pemeriksaan ahli pertanian."
    )
    render_footer()


if __name__ == "__main__":
    main()
