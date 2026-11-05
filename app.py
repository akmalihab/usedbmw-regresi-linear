import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="BMW Price Predictor",
    page_icon="🚗",
    layout="wide",
)

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
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
        return model, preprocessor, label_encoder, metrics
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

st.title("🚗 BMW Price Predictor")
st.subheader("Prediksi harga mobil BMW bekas menggunakan Linear Regression")

# Load data dan model
if not CSV_PATH.exists():
    st.error("Dataset bmw.csv tidak ditemukan.")
    st.stop()

df = load_csv()
model, preprocessor, label_encoder, metrics = load_model_artifacts()

if model is None or preprocessor is None:
    st.error("Model belum tersedia. Jalankan analisis.py atau analisis.ipynb terlebih dahulu.")
    st.stop()

max_year = int(df["year"].max())

# Layout utama
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### Input Spesifikasi")
    
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
    st.markdown("### Hasil Prediksi")
    
    if submitted:
        try:
            # Bangun input
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
            
            # Feature engineering
            input_data["age"] = max_year - input_data["year"] + 1
            
            # Label encode model
            if label_encoder is not None:
                if model_name.strip() in label_encoder.classes_:
                    input_data["model_encoded"] = label_encoder.transform(input_data["model"])
                else:
                    st.warning(f"Model '{model_name}' tidak dikenali, menggunakan nilai default.")
                    input_data["model_encoded"] = 0
                input_data = input_data.drop(columns=["model"])
            
            # Transform dan prediksi
            X_transformed = preprocessor.transform(input_data)
            prediction = model.predict(X_transformed)[0]
            prediction = max(prediction, 0)
            
            # Tampilkan hasil
            st.success("✅ Prediksi Berhasil")
            st.metric(
                label="Estimasi Harga",
                value=f"£{prediction:,.0f}",
                help=f"{model_name} · {year} · {transmission}"
            )
            st.caption(f"📊 {model_name} · {year} · {transmission}")
            
            # Detail tambahan
            with st.expander("Detail Input"):
                st.dataframe(input_data.T, use_container_width=True)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            st.exception(e)
    else:
        st.info("Isi formulir di sebelah kiri dan tekan tombol Prediksi Harga")

# Metrik model
st.divider()
st.markdown("### Performa Model")

if metrics:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("MAE", f"£{metrics.get('MAE', 0):,.0f}")
    col_m2.metric("RMSE", f"£{metrics.get('RMSE', 0):,.0f}")
    col_m3.metric("R²", f"{metrics.get('R2', 0):.4f}")
    col_m4.metric("Adj R²", f"{metrics.get('Adj_R2', 0):.4f}")

# Penjelasan metrik
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

# Info dataset
with st.expander("Informasi Dataset"):
    st.markdown(f"""
    - **Jumlah data:** {len(df):,} mobil BMW bekas
    - **Fitur:** {', '.join(df.columns.tolist())}
    - **Rentang tahun:** {df['year'].min()} - {df['year'].max()}
    - **Rentang harga:** £{df['price'].min():,} - £{df['price'].max():,}
    - **Model BMW:** {df['model'].nunique()} varian
    """)

st.divider()
st.caption("Model: Linear Regression | Dataset: 10,664 mobil BMW bekas (setelah cleaning)")
