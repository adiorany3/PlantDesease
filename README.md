# Plant Disease Detection - Streamlit Online App

Aplikasi Streamlit untuk mendeteksi penyakit tanaman dari gambar daun menggunakan model TensorFlow/Keras.

## File utama

```text
app.py
plant_disease_model.keras
class_names.json
requirements.txt
.streamlit/config.toml
```

## Jumlah kelas

Model ini mendukung **38 kelas** penyakit/sehat tanaman.

Contoh kelas:

```text
Apple___Apple_scab
Apple___Black_rot
Apple___Cedar_apple_rust
Apple___healthy
Blueberry___healthy
Cherry_(including_sour)___Powdery_mildew
Cherry_(including_sour)___healthy
Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
Corn_(maize)___Common_rust_
Corn_(maize)___Northern_Leaf_Blight
Corn_(maize)___healthy
Grape___Black_rot
```

## Jalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Buat repository GitHub baru.
2. Upload semua file dari folder ini ke repository.
3. Buka Streamlit Community Cloud.
4. Klik **New app**.
5. Pilih repository GitHub.
6. Isi **Main file path** dengan:

```text
app.py
```

7. Klik **Deploy**.

## Catatan ukuran file model

Jika GitHub menolak upload file `.keras` karena ukuran terlalu besar, gunakan Git LFS atau deploy di platform yang mendukung file model besar.
