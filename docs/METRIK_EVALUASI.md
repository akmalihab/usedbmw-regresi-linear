# Metrik Evaluasi Model Regresi Linear

Dokumen ini menjelaskan metrik evaluasi yang digunakan untuk mengukur performa model regresi linear dalam memprediksi harga mobil BMW bekas.

## Metrik yang Digunakan

### 1. MAE (Mean Absolute Error)

**Definisi:** Rata-rata dari selisih absolut antara nilai aktual dan nilai prediksi.

**Formula:**
```
MAE = (1/n) × Σ|y_actual - y_predicted|
```

**Nilai Model:** £3,792.85

**Interpretasi:**
- MAE menunjukkan rata-rata kesalahan prediksi dalam satuan yang sama dengan target (£).
- Model kita rata-rata meleset sekitar £3,793 dari harga sebenarnya.
- Metrik ini tidak memberikan bobot lebih pada kesalahan besar, sehingga lebih robust terhadap outlier.
- Dalam konteks bisnis: Jika menggunakan model ini untuk estimasi, perkirakan margin error ±£3,793.

**Kelebihan:**
- Mudah dipahami (dalam satuan £)
- Robust terhadap outlier
- Memberikan gambaran langsung kesalahan rata-rata

**Kekurangan:**
- Tidak membedakan antara over-prediction dan under-prediction
- Tidak sensitif terhadap kesalahan ekstrem

---

### 2. MSE (Mean Squared Error)

**Definisi:** Rata-rata dari kuadrat selisih antara nilai aktual dan nilai prediksi.

**Formula:**
```
MSE = (1/n) × Σ(y_actual - y_predicted)²
```

**Nilai Model:** 30,699,229.25

**Interpretasi:**
- MSE mengkuadratkan kesalahan, sehingga memberikan penalti lebih besar pada prediksi yang meleset jauh.
- Nilai MSE dalam satuan kuadrat (£²), sehingga sulit diinterpretasi langsung.
- Lebih berguna untuk optimisasi matematis daripada interpretasi bisnis.

**Kelebihan:**
- Sensitif terhadap outlier (baik untuk mendeteksi kesalahan besar)
- Fungsi differentiable (baik untuk optimisasi gradient descent)

**Kekurangan:**
- Satuan tidak intuitif (£²)
- Sangat dipengaruhi outlier

---

### 3. RMSE (Root Mean Squared Error)

**Definisi:** Akar kuadrat dari MSE. Mengembalikan metrik ke satuan asli.

**Formula:**
```
RMSE = √MSE = √[(1/n) × Σ(y_actual - y_predicted)²]
```

**Nilai Model:** £5,540.69

**Interpretasi:**
- RMSE merepresentasikan standar deviasi dari kesalahan prediksi.
- Nilai £5,541 berarti sebaran kesalahan prediksi memiliki standar deviasi sekitar £5,541.
- RMSE > MAE (£5,541 vs £3,793) mengindikasikan adanya beberapa prediksi dengan kesalahan besar.
- Dalam konteks probabilistik: ~68% prediksi berada dalam rentang ±£5,541 dari nilai aktual (asumsi distribusi normal).

**Perbandingan dengan MAE:**
- Jika RMSE ≈ MAE → kesalahan terdistribusi merata
- Jika RMSE >> MAE → ada banyak outlier atau kesalahan ekstrem
- Rasio RMSE/MAE = 5,541/3,793 = 1.46 → ada beberapa kesalahan besar, namun tidak terlalu ekstrem

**Kelebihan:**
- Dalam satuan yang sama dengan target (£)
- Lebih sensitif terhadap kesalahan besar
- Umum digunakan dan mudah dibandingkan antar model

**Kekurangan:**
- Lebih dipengaruhi outlier dibanding MAE
- Sulit diinterpretasi secara intuitif

---

### 4. R² (Coefficient of Determination)

**Definisi:** Proporsi variansi dalam variabel target yang dapat dijelaskan oleh model.

**Formula:**
```
R² = 1 - (SS_residual / SS_total)

dimana:
SS_residual = Σ(y_actual - y_predicted)²
SS_total = Σ(y_actual - ȳ)²
```

**Nilai Model:** 0.7536

**Interpretasi:**
- R² = 0.7536 berarti model dapat menjelaskan **75.36%** variasi dalam harga mobil BMW.
- Sisanya 24.64% dipengaruhi oleh faktor-faktor yang tidak termasuk dalam model (kondisi mobil, lokasi, reputasi penjual, dll).
- Rentang nilai: 0 (model tidak menjelaskan apa-apa) hingga 1 (model sempurna).

**Kategori Performa (Rule of Thumb):**
- R² < 0.3 : Performa lemah
- 0.3 ≤ R² < 0.5 : Performa moderate
- 0.5 ≤ R² < 0.7 : Performa baik
- R² ≥ 0.7 : **Performa sangat baik** ← Model kita di sini
- R² > 0.9 : Excellent (atau overfitting jika training data)

**Interpretasi Bisnis:**
Model ini cukup andal untuk:
- Estimasi harga awal dalam listing
- Screening deal yang terlalu mahal/murah
- Benchmark harga market

**TIDAK cocok untuk:**
- Keputusan finansial final (kredit, asuransi)
- Transaksi high-value tanpa verifikasi manual
- Pricing strategy tunggal tanpa expert judgment

**Kelebihan:**
- Mudah dipahami (persentase)
- Independen terhadap skala target
- Standar industri untuk evaluasi model regresi

**Kekurangan:**
- Bisa meningkat hanya dengan menambah fitur (tanpa memperbaiki prediksi)
- Tidak informatif untuk interpretasi error dalam satuan £

---

### 5. Adjusted R² (Adjusted Coefficient of Determination)

**Definisi:** R² yang disesuaikan dengan jumlah prediktor (fitur) dalam model.

**Formula:**
```
Adjusted R² = 1 - [(1 - R²) × (n - 1) / (n - p - 1)]

dimana:
n = jumlah sampel
p = jumlah prediktor/fitur
```

**Nilai Model:** 0.7521

**Interpretasi:**
- Adjusted R² = 0.7521, sangat dekat dengan R² = 0.7536.
- Perbedaan kecil (0.0015) menunjukkan bahwa fitur yang digunakan (13 fitur) relevan dan tidak ada redundansi berlebihan.
- Jika banyak fitur tidak berguna, Adjusted R² akan jauh lebih rendah dari R².

**Kapan Menggunakan Adjusted R² vs R²:**
- **R² biasa:** Untuk membandingkan model dengan jumlah fitur yang sama
- **Adjusted R²:** Untuk membandingkan model dengan jumlah fitur berbeda (mencegah bias penambahan fitur)

**Perbandingan dengan R²:**
- Jika Adj R² ≈ R² → fitur relevan, tidak ada overfitting
- Jika Adj R² << R² → terlalu banyak fitur tidak penting
- Model kita: 0.7521 ≈ 0.7536 → **13 fitur yang dipilih efisien dan relevan**

**Kelebihan:**
- Menghukum penambahan fitur yang tidak meningkatkan performa
- Lebih reliable untuk model dengan banyak prediktor
- Mencegah overfitting dari feature bloating

**Kekurangan:**
- Masih tidak memberikan informasi error dalam satuan £
- Interpretasi sedikit lebih kompleks dibanding R²

---

## Ringkasan Performa Model

| Metrik | Nilai | Kategori |
|--------|-------|----------|
| MAE | £3,792.85 | Baik untuk dataset harga £22,733 (17% error) |
| RMSE | £5,540.69 | Standar deviasi error reasonable |
| R² | 0.7536 | Sangat baik (75.36% variansi dijelaskan) |
| Adjusted R² | 0.7521 | Fitur efisien, tidak ada redundansi |

---

## Analisis Cross-Metric

### 1. MAE vs RMSE
- **Rasio RMSE/MAE:** 1.46
- **Interpretasi:** Ada beberapa prediksi dengan kesalahan besar, tetapi tidak terlalu ekstrem.
- **Rekomendasi:** Investigasi outlier pada residual plot untuk identifikasi pola.

### 2. R² vs Adjusted R²
- **Selisih:** 0.0015 (sangat kecil)
- **Interpretasi:** 13 fitur yang digunakan semuanya relevan.
- **Rekomendasi:** Tidak perlu feature reduction, model sudah efisien.

### 3. Error Relatif
- **MAE/Mean Price:** 3,793 / 22,733 = **16.7%**
- **Interpretasi:** Rata-rata error sekitar 17% dari harga rata-rata.
- **Benchmark:** Untuk harga mobil bekas, error 15-20% dianggap acceptable.

---

## Visualisasi Evaluasi

### 1. Actual vs Predicted Plot
- **Interpretasi:** Titik-titik mendekati garis diagonal y=x menunjukkan prediksi baik.
- **Pattern:** Jika titik tersebar acak di sekitar garis → model unbiased.
- **Red flag:** Jika titik membentuk kurva → hubungan non-linear tidak tertangkap.

### 2. Residual Distribution
- **Ideal:** Distribusi normal dengan mean ≈ 0.
- **Interpretasi:** Distribusi simetris → model tidak bias sistematis.
- **Red flag:** Distribusi skewed → model cenderung over/under-predict.

### 3. Residual vs Predicted
- **Ideal:** Titik tersebar acak tanpa pola (homoskedastisitas).
- **Red flag:** Pola funnel → heteroskedastisitas (variansi tidak konstan).
- **Rekomendasi:** Jika heteroskedastisitas terdeteksi, pertimbangkan transformasi log atau model robust.

---

## Limitasi Model

Meskipun metrik menunjukkan performa baik, perlu diingat:

1. **Model Linear Assumption**
   - Hubungan non-linear tidak tertangkap
   - Interaksi kompleks antar fitur diabaikan

2. **Data Limitation**
   - Hanya fitur kuantitatif (year, mileage, dll)
   - Tidak ada fitur kondisi fisik mobil
   - Tidak ada faktor geografis/musiman

3. **Outlier Sensitivity**
   - RMSE > MAE → beberapa prediksi dengan error besar
   - Perlu investigasi mobil luxury/rare yang harganya anomali

4. **Generalisasi**
   - Model dilatih pada data UK (£)
   - Performa bisa berbeda di market lain

---

## Rekomendasi Penggunaan

### ✅ Cocok Untuk:
- Estimasi harga awal untuk listing
- Screening deal yang mencurigakan (terlalu murah/mahal)
- Benchmark harga untuk negosiasi
- Dashboard internal dealer

### ⚠️ Gunakan dengan Hati-hati:
- Mobil rare/luxury (model mungkin under-trained)
- Kondisi fisik ekstrem (accident, restoration)
- Keputusan finansial >£50k

### ❌ Tidak Cocok Untuk:
- Final pricing tanpa inspeksi fisik
- Underwriting asuransi/kredit (perlu model lebih robust)
- Compliance/regulasi (perlu interpretability tinggi)

---

## Kesimpulan

Model regresi linear ini menunjukkan **performa sangat baik** (R² = 0.7536) dengan error rata-rata £3,793 (17% dari harga rata-rata). Model cocok untuk estimasi harga awal dan screening, namun tetap memerlukan validasi manual untuk transaksi penting.

**Key Takeaways:**
- ✅ 75% variasi harga dapat dijelaskan model
- ✅ Error rata-rata £3,793 reasonable untuk use case estimasi
- ✅ Fitur yang dipilih efisien (Adj R² ≈ R²)
- ⚠️ Ada beberapa prediksi dengan error besar (RMSE > MAE)
- 💡 Potensi peningkatan: model non-linear, feature engineering lanjutan, ensemble methods
