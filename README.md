# Plant Disease Detection - GitHub Model Patch

Patch ini tidak menyertakan file model agar ZIP ringan.

Aplikasi akan mencari model di root repository:

```text
plant_disease_model.keras
```

Jika file lokal tidak valid, misalnya Git LFS pointer, aplikasi akan mencoba mengambil model dari:

```text
https://raw.githubusercontent.com/adiorany3/PlantDesease/main/plant_disease_model.keras
```

Anda juga bisa mengganti sumber model melalui Streamlit Secrets:

```toml
MODEL_URL = "https://direct-download-url/plant_disease_model.keras"
```

## File yang disertakan

```text
app.py
class_names.json
requirements.txt
.python-version
runtime.txt
README.md
.streamlit/config.toml
```

## File yang tidak disertakan

```text
plant_disease_model.keras
```

File model tetap harus ada di GitHub atau tersedia melalui `MODEL_URL`.

## Jumlah kelas

`class_names.json` berisi 38 kelas penyakit/sehat tanaman.

## Catatan penting

`requirements.txt` harus ditulis per baris, bukan satu baris panjang.

Benar:

```text
streamlit>=1.35,<2
tensorflow-cpu==2.15.0
numpy==1.26.4
pillow==10.3.0
h5py==3.10.0
```
