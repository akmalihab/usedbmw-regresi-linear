# Analisis Regresi Linear - Harga Mobil BMW Bekas

Proyek ini menganalisis dataset harga mobil BMW bekas dan membangun model prediksi harga menggunakan Linear Regression.

## Struktur Proyek

```
usedbmw-regresi-linear/
├── data/
│   └── bmw.csv                      # Dataset (10,781 baris, 9 kolom)
├── notebooks/
│   └── analisis.ipynb               # Jupyter Notebook analisis interaktif
├── src/
│   └── train.py                     # Script training model
├── model/
│   ├── linear_regression_model.pkl  # Model terlatih
│   ├── preprocessor.pkl             # Pipeline preprocessing
│   ├── feature_names.pkl            # Nama fitur setelah encoding
│   ├── model_metrics.pkl            # Metrik evaluasi
│   └── label_encoders.pkl           # LabelEncoder
├── plots/
│   └── *.png                        # 14 visualisasi
├── docs/
│   ├── METRIK_EVALUASI.md          # Penjelasan metrik detail
│   └── KESIMPULAN_ANALISIS.md      # Kesimpulan & rekomendasi
├── app.py                           # Dashboard Streamlit
├── README.md                        # Dokumentasi utama
├── CHANGELOG.md                     # Log perubahan
└── requirements.txt                 # Dependencies
```

## Quick Start

### 1. Training Model

#### Opsi A: Jupyter Notebook (Recommended)
```bash
cd notebooks
jupyter notebook analisis.ipynb
```

#### Opsi B: Python Script
```bash
cd src
python train.py
```

Output: 14 plots di `plots/` + 5 model artifacts di `model/`

### 2. Dashboard Streamlit

```bash
streamlit run app.py
```

Akses: `http://localhost:8501`

## Dataset

**File:** `data/bmw.csv`  
**Ukuran:** 10,781 baris → 10,664 (setelah cleaning)

### Kolom

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `model` | str | Model BMW (24 varian) |
| `year` | int | Tahun produksi (1996-2020) |
| `price` | int | Harga (£) - **Target** |
| `transmission` | str | Automatic, Manual, Semi-Auto |
| `mileage` | int | Jarak tempuh (mil) |
| `fuelType` | str | Diesel, Petrol, Hybrid, Electric, Other |
| `tax` | int | Pajak tahunan (£) |
| `mpg` | float | Efisiensi bahan bakar |
| `engineSize` | float | Ukuran mesin (L) |

## Hasil Model

| Metrik | Nilai | Interpretasi |
|--------|-------|--------------|
| **R²** | 0.7536 | 75.36% variansi dijelaskan |
| **Adjusted R²** | 0.7521 | Fitur efisien |
| **MAE** | £3,793 | Error rata-rata |
| **RMSE** | £5,541 | Standar deviasi error |

**Kesimpulan:** Model memiliki performa sangat baik untuk estimasi harga awal. [Detail lengkap →](docs/KESIMPULAN_ANALISIS.md)

## Dokumentasi

- **[Metrik Evaluasi](docs/METRIK_EVALUASI.md)** - Penjelasan MAE, RMSE, R², Adjusted R² dengan interpretasi bisnis
- **[Kesimpulan Analisis](docs/KESIMPULAN_ANALISIS.md)** - Temuan, insight, limitasi, dan rekomendasi
- **[Changelog](CHANGELOG.md)** - Log perubahan proyek

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib jupyter
```

Atau:
```bash
pip install -r requirements.txt
```

## Workflow Analisis

1. **Statistik Deskriptif** - Eksplorasi data, missing values, distribusi
2. **EDA** - 14 visualisasi (distribusi, korelasi, scatter, boxplot)
3. **Preprocessing** - Cleaning, feature engineering (`age`), encoding, scaling
4. **Modeling** - Linear Regression dengan 13 fitur
5. **Evaluasi** - MAE, RMSE, R², residual analysis
6. **Deployment** - Dashboard Streamlit untuk prediksi real-time

## Fitur Dashboard

- Form input spesifikasi mobil
- Prediksi harga real-time
- Metrik performa model
- Penjelasan metrik evaluasi
- Info dataset

## Top Insights

1. **Fuel type** adalah prediktor terkuat (Electric +£12,439, Hybrid +£8,389)
2. **Engine size** setiap 1L → +£4,504 harga
3. **Age** dan **mileage** berkorelasi negatif kuat dengan harga
4. Model menjelaskan **75%** variasi harga (sangat baik)

## Rekomendasi Penggunaan

### ✅ Cocok untuk:
- Estimasi harga awal listing
- Screening deal mencurigakan
- Benchmark negosiasi
- Dashboard internal dealer

### ⚠️ Gunakan dengan validasi:
- Transaksi >£40k
- Mobil rare/luxury
- Kondisi fisik ekstrem

### ❌ Tidak cocok untuk:
- Final pricing tanpa inspeksi
- Underwriting kredit/asuransi
- Keputusan finansial regulasi

## Pengembangan Lanjutan

**Peningkatan akurasi:**
- Random Forest / Gradient Boosting (target R² > 0.85)
- Feature engineering: interaction terms, polynomial
- Data enrichment: kondisi fisik, lokasi, service history

**Deployment:**
- REST API untuk integrasi sistem
- Model monitoring & retraining berkala
- A/B testing model baru vs lama

## Lisensi

Dataset dan kode untuk keperluan edukasi.

---

**Terakhir diupdate:** 2026-11-05  
**Model Version:** 1.0
