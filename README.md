# Plant Disease Detection - Streamlit Ringan

Paket ini tidak menyertakan file model agar ukuran ZIP/repository lebih ringan.

## Fitur

- Upload gambar daun dari file.
- Ambil foto langsung dari kamera.
- UI lebih rapi.
- Branding/emblem Streamlit pada header/footer disamarkan melalui CSS.
- Model dapat dimuat dari:
  - file lokal `plant_disease_model.h5`, atau
  - `MODEL_URL` di Streamlit Secrets.

## Jumlah Kelas

Model menggunakan `class_names.json` berisi 38 kelas penyakit/sehat tanaman.

## Struktur File

```text
app.py
class_names.json
requirements.txt
.python-version
.streamlit/config.toml
.streamlit/secrets.toml.example
README.md
```

## Cara Deploy ke Streamlit Community Cloud

1. Upload semua file project ke root repository GitHub.
2. Deploy app dengan Python 3.11.
3. Main file path: `app.py`.
4. Upload model `.h5` ke penyimpanan publik.
5. Buka **Manage app → Settings → Secrets**.
6. Isi:

```toml
MODEL_URL = "https://direct-download-url/plant_disease_model.h5"
```

7. Save dan reboot app.

## Alternatif

Jika tetap ingin model berada di repository, upload file model dengan nama:

```text
plant_disease_model.h5
```

dan letakkan sejajar dengan `app.py`.

## Jalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```
