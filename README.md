# Plant Disease Detection - Streamlit App

Aplikasi Streamlit untuk deteksi penyakit tanaman dari gambar daun.

## Penting untuk Streamlit Community Cloud

Saat deploy, pilih **Python 3.11** pada **Advanced settings**.
TensorFlow 2.15 tidak kompatibel dengan Python 3.14, sehingga app akan gagal install dependency jika Python Cloud dibiarkan memakai 3.14.

Jika app sudah terlanjur dibuat dengan Python 3.14, hapus app di Streamlit Cloud lalu deploy ulang dengan Python 3.11.

## File utama

```text
app.py
plant_disease_model.keras
class_names.json
requirements.txt
.streamlit/config.toml
```

## Cara run lokal

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Di Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
