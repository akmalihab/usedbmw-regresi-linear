# Changelog

Log semua perubahan signifikan dalam proyek ini.

## [2026-11-05] - Iterasi 3 (Restruktursasi)

### Restrukturisasi Proyek
- **Struktur folder baru:**
  ```
  ├── data/           # Dataset
  ├── notebooks/      # Jupyter notebooks
  ├── src/            # Source code (train.py)
  ├── model/          # Model artifacts
  ├── plots/          # Visualisasi
  ├── docs/           # Dokumentasi
  └── app.py          # Dashboard Streamlit (root untuk akses mudah)
  ```

### Dokumentasi Baru
- **`docs/METRIK_EVALUASI.md`**
  - Penjelasan detail MAE, RMSE, R², Adjusted R²
  - Formula, interpretasi, kelebihan/kekurangan
  - Analisis cross-metric
  - Rekomendasi penggunaan berdasarkan metrik
  
- **`docs/KESIMPULAN_ANALISIS.md`**
  - Executive summary
  - Temuan EDA dan insight bisnis
  - Analisis koefisien model
  - Limitasi dan rekomendasi
  - Business impact projection
  - ROI estimate

- **`requirements.txt`**
  - List dependencies untuk easy setup

### Update File
- **README.md** - Update struktur dan quick start guide
- **src/train.py** - Path update ke `../data/bmw.csv`
- **app.py** - Pindah ke root untuk akses mudah, path update ke `data/`, `model/`
- **notebooks/analisis.ipynb** - Path update ke `../data/`, `../plots/`, `../model/`

## [2026-11-05] - Iterasi 2

### Perubahan Mayor
- **Konversi analisis.py ke Jupyter Notebook**
  - File baru: `analisis.ipynb`
  - Format interaktif untuk eksplorasi data
  - Semua cell terstruktur per bagian analisis
  
- **Redesign Dashboard Streamlit**
  - Dari 5 halaman multi-page → 1 halaman minimalis
  - Fokus pada prediksi harga (use case utama)
  - UI lebih bersih, kurangi emoji
  - Layout 2 kolom: input form (kiri) | hasil prediksi (kanan)
  
- **Penjelasan Metrik Evaluasi**
  - Tambah expander "Penjelasan Metrik Evaluasi" di dashboard
  - Penjelasan detail MAE, RMSE, R², Adjusted R²
  - Interpretasi dalam konteks bisnis

### Perbaikan
- Fix bug prediksi: kolom `model` di-rename ke `model_encoded` setelah label encoding
- Hilangkan dropdown year jika age sudah ada di feature names

## [2026-11-05] - Iterasi 1 (Initial)

### Fitur Awal
- **Script Analisis** (`analisis.py`)
  - Statistik deskriptif
  - EDA dengan 14 visualisasi
  - Preprocessing (cleaning, encoding, scaling)
  - Model Linear Regression
  - Evaluasi (MAE, RMSE, R², Adj R²)
  - Simpan model artifacts
  
- **Dashboard Streamlit Multi-Page** (`app.py`)
  - Halaman 1: Statistik Deskriptif
  - Halaman 2: EDA (tabs untuk kategori plot)
  - Halaman 3: Preprocessing
  - Halaman 4: Model & Evaluasi
  - Halaman 5: Prediksi Harga
  - Tema BMW (blue accent #1C69D4)
  
- **Dataset**
  - `bmw.csv`: 10,781 mobil BMW bekas
  - 9 kolom (model, year, price, transmission, mileage, fuelType, tax, mpg, engineSize)
  
- **Model Artifacts**
  - `linear_regression_model.pkl`
  - `preprocessor.pkl`
  - `feature_names.pkl`
  - `model_metrics.pkl`
  - `label_encoders.pkl`
  
- **Visualisasi** (14 file PNG di `plots/`)
  - Distribusi, korelasi, scatter plots, boxplots
  - Evaluation plots (actual vs predicted, residual analysis)

### Hasil Model
- **R² = 0.7536** (75.36% variansi dijelaskan)
- **MAE = £3,793** (rata-rata error)
- **RMSE = £5,541** (standar deviasi error)
- **Adjusted R² = 0.7521**
- **Data:** 8,531 training / 2,133 testing (80:20 split)

---

## Format

Changelog mengikuti format berikut:
- **[YYYY-MM-DD]** - Nama versi/iterasi
- **Kategori:** Added, Changed, Fixed, Removed, Deprecated
- **Deskripsi singkat** perubahan
