import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="BMW Price Predictor",
    layout="wide",
)

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
PLOTS_DIR = BASE / "plots"
CSV_PATH = BASE / "data" / "bmw.csv"

@st.cache_data
def load_csv():
    return pd.read_csv(CSV_PATH)

@st.cache_data
def load_pkl(path):
    return joblib.load(path)

def load_model_artifacts():
    """Muat semua artefak model"""
    try:
        model = load_pkl(str(MODEL_DIR / "linear_regression_model.pkl"))
        preprocessor = load_pkl(str(MODEL_DIR / "preprocessor.pkl"))
        label_encoder = load_pkl(str(MODEL_DIR / "label_encoders.pkl"))
        metrics = load_pkl(str(MODEL_DIR / "model_metrics.pkl"))
        feature_names = load_pkl(str(MODEL_DIR / "feature_names.pkl"))
        return model, preprocessor, label_encoder, metrics, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

st.title("BMW Price Predictor")
st.subheader("Prediksi harga mobil BMW bekas menggunakan Linear Regression")

# Load data dan model
if not CSV_PATH.exists():
    st.error("Dataset bmw.csv tidak ditemukan.")
    st.stop()

df = load_csv()
model, preprocessor, label_encoder, metrics, feature_names = load_model_artifacts()

if model is None or preprocessor is None:
    st.error("Model belum tersedia. Jalankan analisis.py atau analisis.ipynb terlebih dahulu.")
    st.stop()

max_year = int(df["year"].max())

# Tabs
tab1, tab2 = st.tabs(["Prediksi Harga", "Analisis Data"])

# ==================== TAB 1: PREDIKSI ====================
with tab1:
    st.markdown("### Prediksi Harga Mobil BMW Bekas")
    
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown("#### Input Spesifikasi")
        
        with st.form("prediction_form"):
            model_name = st.selectbox(
                "Model",
                sorted(df["model"].str.strip().unique()),
                help="Pilih model BMW"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                year = st.number_input("Tahun", min_value=2000, max_value=2025, value=2018, step=1)
            with col2:
                transmission = st.selectbox("Transmisi", ["Automatic", "Manual", "Semi-Auto"])
            
            col3, col4 = st.columns(2)
            with col3:
                mileage = st.number_input("Mileage (mil)", min_value=0, max_value=300000, value=20000, step=1000)
            with col4:
                fuel_type = st.selectbox("Bahan Bakar", ["Diesel", "Petrol", "Hybrid", "Electric", "Other"])
            
            col5, col6 = st.columns(2)
            with col5:
                tax = st.number_input("Pajak Tahunan (£)", min_value=0, max_value=600, value=145, step=5)
            with col6:
                mpg = st.number_input("MPG", min_value=0.0, max_value=300.0, value=55.0, step=1.0)
            
            engine_size = st.number_input("Ukuran Mesin (L)", min_value=0.0, max_value=7.0, value=2.0, step=0.1)
            
            submitted = st.form_submit_button("Prediksi Harga")
    
    with col_right:
        st.markdown("#### Hasil Prediksi")
        
        if submitted:
            try:
                input_data = pd.DataFrame([{
                    "model": model_name.strip(),
                    "year": year,
                    "transmission": transmission,
                    "mileage": mileage,
                    "fuelType": fuel_type,
                    "tax": tax,
                    "mpg": mpg,
                    "engineSize": engine_size,
                }])
                
                input_data["age"] = max_year - input_data["year"] + 1
                
                if label_encoder is not None:
                    if model_name.strip() in label_encoder.classes_:
                        input_data["model_encoded"] = label_encoder.transform(input_data["model"])
                    else:
                        st.warning(f"Model '{model_name}' tidak dikenali, menggunakan nilai default.")
                        input_data["model_encoded"] = 0
                    input_data = input_data.drop(columns=["model"])
                
                X_transformed = preprocessor.transform(input_data)
                prediction = model.predict(X_transformed)[0]
                prediction = max(prediction, 0)
                
                st.success("Prediksi Berhasil")
                st.metric(
                    label="Estimasi Harga",
                    value=f"£{prediction:,.0f}",
                    help=f"{model_name} · {year} · {transmission}"
                )
                st.caption(f"{model_name} · {year} · {transmission}")
                
                with st.expander("Detail Input"):
                    st.dataframe(input_data.T, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                st.exception(e)
        else:
            st.info("Isi formulir di sebelah kiri dan tekan tombol Prediksi Harga")
    
    st.divider()
    st.markdown("### Performa Model")
    
    if metrics:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("MAE", f"£{metrics.get('MAE', 0):,.0f}")
        col_m2.metric("RMSE", f"£{metrics.get('RMSE', 0):,.0f}")
        col_m3.metric("R²", f"{metrics.get('R2', 0):.4f}")
        col_m4.metric("Adj R²", f"{metrics.get('Adj_R2', 0):.4f}")
    
    with st.expander("Penjelasan Metrik Evaluasi"):
        st.markdown("""
        #### Mean Absolute Error (MAE)
        Rata-rata dari selisih absolut antara nilai aktual dan prediksi. Metrik ini memberikan gambaran kesalahan rata-rata dalam satuan yang sama dengan target (£).
        
        **Interpretasi:** Semakin rendah MAE, semakin baik. Nilai MAE £3,793 berarti rata-rata prediksi meleset sekitar £3,793 dari harga sebenarnya.
        
        #### Root Mean Squared Error (RMSE)
        Akar dari rata-rata kuadrat kesalahan. RMSE lebih sensitif terhadap outlier dibanding MAE karena kesalahan dikuadratkan.
        
        **Interpretasi:** RMSE £5,541 menunjukkan deviasi standar dari kesalahan prediksi. Nilai lebih besar dari MAE mengindikasikan adanya beberapa prediksi dengan kesalahan besar.
        
        #### R² (Coefficient of Determination)
        Proporsi variansi dalam variabel target yang dapat dijelaskan oleh model. Rentang nilai: 0 sampai 1.
        
        **Interpretasi:** R² = 0.7536 berarti model dapat menjelaskan 75.36% variasi harga mobil BMW. Sisanya (24.64%) dipengaruhi faktor lain yang tidak termasuk dalam model.
        
        #### Adjusted R²
        Versi R² yang disesuaikan dengan jumlah fitur dalam model. Lebih reliable untuk model dengan banyak variabel.
        
        **Interpretasi:** Adj R² = 0.7521 menunjukkan performa model setelah memperhitungkan kompleksitas. Nilai yang sangat dekat dengan R² mengindikasikan fitur yang digunakan relevan.
        
        ---
        
        **Kesimpulan:** Model memiliki akurasi yang baik (R² > 0.75) dengan rata-rata kesalahan sekitar £3,793. Model ini cocok untuk estimasi harga awal, namun untuk keputusan finansial penting sebaiknya dikombinasikan dengan penilaian manual.
        """)
    
    with st.expander("Informasi Dataset"):
        st.markdown(f"""
        - **Jumlah data:** {len(df):,} mobil BMW bekas
        - **Fitur:** {', '.join(df.columns.tolist())}
        - **Rentang tahun:** {df['year'].min()} - {df['year'].max()}
        - **Rentang harga:** £{df['price'].min():,} - £{df['price'].max():,}
        - **Model BMW:** {df['model'].nunique()} varian
        """)

# ==================== TAB 2: ANALISIS ====================
with tab2:
    st.markdown("### Analisis Data & Model")
    
    # Section 1: Statistik Deskriptif
    st.markdown("#### 1. Statistik Deskriptif")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data", f"{len(df):,}", help="Jumlah baris setelah cleaning")
    with col2:
        st.metric("Fitur", len(df.columns), help="Jumlah kolom dalam dataset")
    with col3:
        st.metric("Model BMW", df['model'].nunique(), help="Jumlah varian model")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.expander("Statistik Numerik", expanded=False):
            st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        with st.expander("Statistik Kategorikal", expanded=False):
            st.dataframe(df.describe(include='object'), use_container_width=True)
    
    st.divider()
    
    # Section 2: Visualisasi EDA
    st.markdown("#### 2. Exploratory Data Analysis")
    
    viz_tabs = st.tabs([
        "Distribusi", 
        "Korelasi", 
        "Scatter Plots", 
        "Boxplots", 
        "Pairplot"
    ])
    
    with viz_tabs[0]:
        st.markdown("##### Distribusi Harga")
        col1, col2 = st.columns([1.2, 1], gap="large")
        with col1:
            if (PLOTS_DIR / "distribusi_harga.png").exists():
                st.image(str(PLOTS_DIR / "distribusi_harga.png"), use_container_width=True)
                st.caption("Histogram menunjukkan distribusi harga dengan kde curve")
        with col2:
            if (PLOTS_DIR / "boxplot_harga.png").exists():
                st.image(str(PLOTS_DIR / "boxplot_harga.png"), use_container_width=True)
                st.caption("Boxplot menampilkan outlier dan statistik 5-number summary")
        
        st.markdown("---")
        st.markdown("##### Distribusi Fitur Numerik")
        if (PLOTS_DIR / "distribusi_fitur_numerik.png").exists():
            st.image(str(PLOTS_DIR / "distribusi_fitur_numerik.png"), use_container_width=True)
            st.caption("Distribusi 5 fitur numerik: year, mileage, tax, mpg, engineSize")
    
    with viz_tabs[1]:
        st.markdown("##### Korelasi Antar Fitur Numerik")
        
        # Centered with padding
        col = st.columns([0.5, 2, 0.5])
        with col[1]:
            if (PLOTS_DIR / "korelasi_heatmap.png").exists():
                st.image(str(PLOTS_DIR / "korelasi_heatmap.png"), use_container_width=True)
                st.caption("Heatmap korelasi Pearson - nilai mendekati 1/-1 menunjukkan korelasi kuat")
        
        st.markdown("**Insight:**")
        st.markdown("""
        - **year** dan **age** berkorelasi sempurna negatif (-1.00) - expected karena age = max_year - year + 1
        - **price** berkorelasi positif kuat dengan **year** (0.64)
        - **price** berkorelasi negatif kuat dengan **mileage** (-0.44) dan **age** (-0.64)
        - **mpg** dan **engineSize** berkorelasi negatif moderat (-0.55)
        """)
    
    with viz_tabs[2]:
        st.markdown("##### Hubungan Fitur dengan Harga")
        
        imgs = [
            ("scatter_mileage_price.png", "Mileage vs Price - Korelasi negatif: semakin tinggi mileage, harga turun"),
            ("scatter_year_price.png", "Year vs Price - Korelasi positif: mobil lebih baru = harga lebih tinggi"),
            ("scatter_enginesize_price.png", "Engine Size vs Price - Korelasi positif: mesin besar = harga tinggi")
        ]
        
        # Display first 2 in row
        cols = st.columns(2, gap="large")
        for j in range(2):
            if j < len(imgs):
                img_file, caption = imgs[j]
                if (PLOTS_DIR / img_file).exists():
                    with cols[j]:
                        st.image(str(PLOTS_DIR / img_file), use_container_width=True)
                        st.caption(caption)
        
        # Display 3rd one centered
        if len(imgs) > 2:
            img_file, caption = imgs[2]
            if (PLOTS_DIR / img_file).exists():
                cols = st.columns([0.5, 2, 0.5])
                with cols[1]:
                    st.image(str(PLOTS_DIR / img_file), use_container_width=True)
                    st.caption(caption)
    
    with viz_tabs[3]:
        st.markdown("##### Harga Berdasarkan Kategori")
        
        st.markdown("###### Transmission & Fuel Type")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            if (PLOTS_DIR / "boxplot_transmission_price.png").exists():
                st.image(str(PLOTS_DIR / "boxplot_transmission_price.png"), use_container_width=True)
                st.caption("Harga berdasarkan jenis transmisi - Automatic & Semi-Auto cenderung lebih mahal")
        with col2:
            if (PLOTS_DIR / "boxplot_fueltype_price.png").exists():
                st.image(str(PLOTS_DIR / "boxplot_fueltype_price.png"), use_container_width=True)
                st.caption("Harga berdasarkan bahan bakar - Electric & Hybrid paling mahal")
        
        st.markdown("---")
        st.markdown("###### Model BMW (Top 10)")
        if (PLOTS_DIR / "boxplot_model_price.png").exists():
            st.image(str(PLOTS_DIR / "boxplot_model_price.png"), use_container_width=True)
            st.caption("Harga berdasarkan model - diurutkan dari median tertinggi ke terendah")
    
    with viz_tabs[4]:
        st.markdown("##### Pairplot Fitur Numerik")
        
        with st.expander("Tentang Pairplot", expanded=False):
            st.markdown("""
            Pairplot menampilkan hubungan antar **semua pasangan fitur numerik**:
            - Diagonal: Distribusi univariate (KDE)
            - Off-diagonal: Scatter plot untuk melihat pola hubungan
            - Berguna untuk deteksi korelasi visual dan outlier
            """)
        
        if (PLOTS_DIR / "pairplot.png").exists():
            st.image(str(PLOTS_DIR / "pairplot.png"), use_container_width=True)
            st.caption("Pairplot 6x6 grid: year, price, mileage, tax, mpg, engineSize")
    
    st.divider()
    
    # Section 3: Preprocessing
    st.markdown("#### 3. Preprocessing")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        with st.expander("Langkah Preprocessing", expanded=True):
            st.markdown("""
            **1. Data Cleaning**
            - Hapus missing values (jika ada)
            - Hapus duplikat (117 baris)
            - Data final: 10,664 baris
            
            **2. Feature Engineering**
            - Buat fitur `age` = 2020 - year + 1
            - Label encoding untuk kolom `model` (24 kelas)
            """)
    
    with col2:
        with st.expander("Encoding & Split", expanded=True):
            st.markdown("""
            **3. Encoding & Scaling**
            - Fitur numerik: StandardScaler
            - Fitur kategorikal: OneHotEncoder (drop first)
            - Total fitur setelah encoding: 13
            
            **4. Train-Test Split**
            - Training: 8,531 baris (80%)
            - Testing: 2,133 baris (20%)
            - Random state: 42
            """)
    
    if feature_names:
        with st.expander("Daftar Fitur Model", expanded=False):
            st.write(f"**Total: {len(feature_names)} fitur**")
            
            # Display as columns for better readability
            num_cols = 3
            cols = st.columns(num_cols)
            for idx, feat in enumerate(feature_names):
                with cols[idx % num_cols]:
                    st.markdown(f"• `{feat}`")
    
    st.divider()
    
    # Section 4: Evaluasi Model
    st.markdown("#### 4. Evaluasi Model")
    
    if metrics:
        st.markdown("##### Metrik Performa")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"£{metrics.get('MAE', 0):,.0f}", help="Mean Absolute Error")
        col2.metric("RMSE", f"£{metrics.get('RMSE', 0):,.0f}", help="Root Mean Squared Error")
        col3.metric("R²", f"{metrics.get('R2', 0):.4f}", help="Coefficient of Determination")
        col4.metric("Adjusted R²", f"{metrics.get('Adj_R2', 0):.4f}", help="Adjusted R²")
    
    st.markdown("---")
    st.markdown("##### Visualisasi Evaluasi")
    
    eval_imgs = [
        ("actual_vs_predicted.png", "Actual vs Predicted - Points mendekati diagonal menunjukkan prediksi akurat"),
        ("residual_plot.png", "Residual Plot - Residual acak sekitar 0 = model baik"),
        ("residual_distribution.png", "Residual Distribution - Distribusi normal = asumsi linear regression terpenuhi"),
        ("residual_qq_plot.png", "QQ Plot - Points mengikuti garis diagonal = residual berdistribusi normal")
    ]
    
    # Display in 2x2 grid
    for i in range(0, len(eval_imgs), 2):
        cols = st.columns(2, gap="large")
        for j in range(2):
            idx = i + j
            if idx < len(eval_imgs):
                img_file, caption = eval_imgs[idx]
                if (PLOTS_DIR / img_file).exists():
                    with cols[j]:
                        st.image(str(PLOTS_DIR / img_file), use_container_width=True)
                        st.caption(caption)
    
    with st.expander("Interpretasi Hasil", expanded=True):
        if metrics:
            st.markdown(f"""
            #### Performa Model
            
            Model Linear Regression mencapai **R² = {metrics.get('R2', 0):.4f}**, yang berarti:
            - Model menjelaskan **{metrics.get('R2', 0)*100:.2f}%** variasi harga mobil BMW
            - Sisanya ({(1-metrics.get('R2', 0))*100:.2f}%) dipengaruhi faktor lain diluar model
            
            **Error Metrics:**
            - **MAE = £{metrics.get('MAE', 0):,.0f}**: Rata-rata prediksi meleset sekitar £{metrics.get('MAE', 0):,.0f}
            - **RMSE = £{metrics.get('RMSE', 0):,.0f}**: Standar deviasi error (lebih besar dari MAE → ada beberapa prediksi dengan error besar)
            
            **Adjusted R² = {metrics.get('Adj_R2', 0):.4f}** sangat dekat dengan R², menunjukkan fitur yang digunakan efisien dan relevan.
            
            #### Kesimpulan
            Model memiliki performa **sangat baik** untuk kasus regresi dengan R² > 0.75. Cocok untuk:
            - Estimasi harga awal listing
            - Screening deal mencurigakan  
            - Benchmark negosiasi
            
            **Limitasi:**
            - Tidak memperhitungkan kondisi fisik mobil
            - Perlu validasi manual untuk mobil luxury/rare
            - Tidak cocok untuk keputusan finansial final tanpa inspeksi
            """)

st.divider()
st.caption("Model: Linear Regression | Dataset: 10,664 mobil BMW bekas (setelah cleaning) | Dashboard v2.0")
