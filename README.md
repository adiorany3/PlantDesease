# Plant Disease Detection - Theme and Insights Patch

Patch ini tidak menyertakan file model agar ZIP tetap ringan.

## Perubahan

- Tampilan dibuat mudah terbaca pada tema terang maupun gelap.
- CSS menggunakan variable warna adaptif dan `prefers-color-scheme`.
- Tidak memaksa background putih, sehingga lebih aman untuk dark mode.
- Insight model ditambahkan:
  - jumlah tanaman
  - jumlah kelas
  - jumlah kelas penyakit
  - jumlah kelas sehat
  - tanaman dengan variasi kelas terbanyak
- Insight hasil prediksi ditambahkan:
  - interpretasi confidence tinggi/sedang/rendah
  - status sehat/berpenyakit
  - selisih prediksi pertama dan kedua
- Footer custom tetap tersedia:

```text
Developed by Galuh Adi Insani, training with Kaggle
```

Link Kaggle:

```text
https://www.kaggle.com/code/adioranye/plant-disease-detection-by-adioranye
```

## File model tidak disertakan

Model tetap harus ada di root repository GitHub:

```text
plant_disease_model.keras
```

## Statistik class_names

- Tanaman: 14
- Total kelas: 38
- Kelas penyakit: 26
- Kelas sehat: 12
