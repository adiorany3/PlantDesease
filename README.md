# Plant Disease Detection - Responsive Footer Patch

Patch ini tidak menyertakan file model agar ZIP ringan.

## Perubahan

- Layout dibuat lebih responsif untuk desktop dan mobile.
- Hero section, card, progress, metric, dan footer dibuat lebih rapi.
- Footer custom ditambahkan:

```text
Developed by Galuh Adi Insani, training with Kaggle
```

Link Kaggle:

```text
https://www.kaggle.com/code/adioranye/plant-disease-detection-by-adioranye
```

- Upload gambar dari file.
- Ambil foto langsung dari kamera.
- Header/footer/menu bawaan Streamlit disamarkan.
- Tetap mencari model di root repository:

```text
plant_disease_model.keras
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
