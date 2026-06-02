# Plant Disease Detection - Streamlit Patch

Patch ini **tidak menyertakan file model** agar ZIP ringan.

Aplikasi akan mencari model yang sudah ada di root repository GitHub dengan nama:

```text
plant_disease_model.keras
```

Jika file tersebut ternyata berisi format HDF5 walaupun ekstensinya `.keras`, aplikasi akan otomatis menyalinnya sementara sebagai `.h5` agar dapat dibaca TensorFlow/Keras.

## File yang disertakan

```text
app.py
class_names.json
requirements.txt
.python-version
.streamlit/config.toml
README.md
```

## File model yang harus tetap ada di GitHub

```text
plant_disease_model.keras
```

Letakkan sejajar dengan `app.py`.

## Fitur

- UI lebih rapi.
- Header/footer/menu bawaan Streamlit disamarkan.
- Upload gambar dari file.
- Ambil foto langsung dari kamera.
- Top 5 hasil prediksi.
- Daftar kelas tanaman/penyakit.

## Jumlah kelas

`class_names.json` berisi 38 kelas.

## Deploy

Gunakan Python 3.11 di Streamlit Community Cloud.

```bash
streamlit run app.py
```
