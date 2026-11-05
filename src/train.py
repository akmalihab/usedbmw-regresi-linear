"""
Analisis Regresi Linear - Dataset Harga Mobil BMW Bekas
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings('ignore')

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('seaborn-whitegrid')

PLOTS_DIR = '../plots'
MODEL_DIR = '../model'
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 1. STATISTIK DESKRIPTIF
# ============================================================
print("=" * 70)
print("1. STATISTIK DESKRIPTIF")
print("=" * 70)

df = pd.read_csv('../data/bmw.csv')
df.columns = df.columns.str.strip()
df['model'] = df['model'].str.strip()

print(f"\nUkuran dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"\nTipe data:\n{df.dtypes}")
print(f"\nInfo dataset:")
df.info()

print(f"\n--- Statistik Deskriptif (Numerik) ---")
print(df.describe().to_string())

print(f"\n--- Statistik Deskriptif (Kategorikal) ---")
print(df.describe(include='object').to_string())

for col in ['model', 'transmission', 'fuelType']:
    print(f"\n--- Value Counts: {col} ---")
    print(df[col].value_counts().to_string())

print(f"\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.sum() > 0 else "Tidak ada missing values.")

n_dup = df.duplicated().sum()
print(f"\nJumlah duplikat: {n_dup}")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS & VISUALISASI
# ============================================================
print("\n" + "=" * 70)
print("2. EXPLORATORY DATA ANALYSIS & VISUALISASI")
print("=" * 70)

# Distribusi harga
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['price'], kde=True, bins=50, ax=ax, color='steelblue')
ax.set_title('Distribusi Harga Mobil BMW Bekas', fontsize=14)
ax.set_xlabel('Harga (£)')
ax.set_ylabel('Frekuensi')
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/distribusi_harga.png', dpi=150)
plt.close(fig)
print("Saved: distribusi_harga.png")

# Boxplot harga
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x=df['price'], ax=ax, color='steelblue')
ax.set_title('Boxplot Harga Mobil BMW Bekas', fontsize=14)
ax.set_xlabel('Harga (£)')
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/boxplot_harga.png', dpi=150)
plt.close(fig)
print("Saved: boxplot_harga.png")

# Korelasi heatmap
num_cols = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, square=True)
ax.set_title('Heatmap Korelasi Fitur Numerik', fontsize=14)
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/korelasi_heatmap.png', dpi=150)
plt.close(fig)
print("Saved: korelasi_heatmap.png")

# Scatter plots
scatter_configs = [
    ('mileage', 'price', 'Jarak Tempuh vs Harga', 'scatter_mileage_price.png'),
    ('year', 'price', 'Tahun vs Harga', 'scatter_year_price.png'),
    ('engineSize', 'price', 'Ukuran Mesin vs Harga', 'scatter_enginesize_price.png'),
]
for xcol, ycol, title, fname in scatter_configs:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[xcol], df[ycol], alpha=0.3, s=10, color='steelblue')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/{fname}', dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")

# Boxplot transmission vs price
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, x='transmission', y='price', ax=ax, palette='Set2')
ax.set_title('Harga Berdasarkan Jenis Transmisi', fontsize=14)
ax.set_xlabel('Transmisi')
ax.set_ylabel('Harga (£)')
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/boxplot_transmission_price.png', dpi=150)
plt.close(fig)
print("Saved: boxplot_transmission_price.png")

# Boxplot fuelType vs price
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, x='fuelType', y='price', ax=ax, palette='Set2')
ax.set_title('Harga Berdasarkan Jenis Bahan Bakar', fontsize=14)
ax.set_xlabel('Jenis Bahan Bakar')
ax.set_ylabel('Harga (£)')
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/boxplot_fueltype_price.png', dpi=150)
plt.close(fig)
print("Saved: boxplot_fueltype_price.png")

# Boxplot top 10 models vs price
top10 = df['model'].value_counts().head(10).index
df_top10 = df[df['model'].isin(top10)]
fig, ax = plt.subplots(figsize=(14, 7))
order = df_top10.groupby('model')['price'].median().sort_values(ascending=False).index
sns.boxplot(data=df_top10, x='model', y='price', order=order, ax=ax, palette='Set3')
ax.set_title('Harga Berdasarkan Model (Top 10)', fontsize=14)
ax.set_xlabel('Model')
ax.set_ylabel('Harga (£)')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/boxplot_model_price.png', dpi=150)
plt.close(fig)
print("Saved: boxplot_model_price.png")

# Pairplot
print("Membuat pairplot (mungkin agak lama)...")
g = sns.pairplot(df[num_cols], diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot Fitur Numerik', y=1.02, fontsize=14)
g.figure.tight_layout()
g.figure.savefig(f'{PLOTS_DIR}/pairplot.png', dpi=120)
plt.close(g.figure)
print("Saved: pairplot.png")

# Distribusi fitur numerik
feat_num = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, col in enumerate(feat_num):
    sns.histplot(df[col], kde=True, bins=40, ax=axes[i], color='steelblue')
    axes[i].set_title(f'Distribusi {col}', fontsize=12)
if len(feat_num) < len(axes):
    for j in range(len(feat_num), len(axes)):
        fig.delaxes(axes[j])
fig.suptitle('Distribusi Fitur Numerik', fontsize=14)
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/distribusi_fitur_numerik.png', dpi=150)
plt.close(fig)
print("Saved: distribusi_fitur_numerik.png")

# ============================================================
# 3. PREPROCESSING
# ============================================================
print("\n" + "=" * 70)
print("3. PREPROCESSING")
print("=" * 70)

# Handle missing
if df.isnull().sum().sum() > 0:
    df = df.dropna()
    print(f"Missing values dihapus. Ukuran baru: {df.shape}")
else:
    print("Tidak ada missing values.")

# Remove duplicates
if n_dup > 0:
    df = df.drop_duplicates()
    print(f"Duplikat dihapus. Ukuran baru: {df.shape}")
else:
    print("Tidak ada duplikat.")

# Feature engineering: age
max_year = df['year'].max()
df['age'] = max_year - df['year'] + 1
print(f"Fitur baru 'age' dibuat. Tahun terbaru={max_year}, age = {max_year} - year + 1")
print(df[['year', 'age']].head().to_string())

# Label encoding for model
le_model = LabelEncoder()
df['model_encoded'] = le_model.fit_transform(df['model'])
print(f"\nLabel Encoding 'model': {len(le_model.classes_)} kelas")

# Define features
target = 'price'
cat_features = ['transmission', 'fuelType']
num_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize', 'age', 'model_encoded']

X = df[num_features + cat_features]
y = df[target]

# Preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set: {X_train.shape[0]} baris")
print(f"Test set:  {X_test.shape[0]} baris")

# Fit preprocessor
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Get feature names after encoding
ohe_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
feature_names = num_features + ohe_names
print(f"Total fitur setelah encoding: {len(feature_names)}")
print(f"Nama fitur: {feature_names}")

# ============================================================
# 4. PEMODELAN REGRESI LINEAR
# ============================================================
print("\n" + "=" * 70)
print("4. PEMODELAN REGRESI LINEAR")
print("=" * 70)

model = LinearRegression()
model.fit(X_train_proc, y_train)

print(f"\nIntercept: {model.intercept_:.2f}")
print(f"\nKoefisien:")
for name, coef in zip(feature_names, model.coef_):
    print(f"  {name:25s} : {coef:>12.4f}")

y_pred = model.predict(X_test_proc)

# ============================================================
# 5. EVALUASI MODEL
# ============================================================
print("\n" + "=" * 70)
print("5. EVALUASI MODEL")
print("=" * 70)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
n, p = X_test_proc.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

metrics = {
    'MAE': mae,
    'MSE': mse,
    'RMSE': rmse,
    'R2': r2,
    'Adj_R2': adj_r2,
}

print(f"\n  MAE          : {mae:>12.2f}")
print(f"  MSE          : {mse:>14.2f}")
print(f"  RMSE         : {rmse:>12.2f}")
print(f"  R²           : {r2:>12.4f}")
print(f"  Adjusted R²  : {adj_r2:>12.4f}")

# Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Garis Ideal')
ax.set_title('Harga Aktual vs Prediksi', fontsize=14)
ax.set_xlabel('Harga Aktual (£)')
ax.set_ylabel('Harga Prediksi (£)')
ax.legend()
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/actual_vs_predicted.png', dpi=150)
plt.close(fig)
print("Saved: actual_vs_predicted.png")

# Residual distribution
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=50, ax=ax, color='steelblue')
ax.axvline(0, color='red', linestyle='--')
ax.set_title('Distribusi Residual', fontsize=14)
ax.set_xlabel('Residual (Aktual - Prediksi)')
ax.set_ylabel('Frekuensi')
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/residual_distribution.png', dpi=150)
plt.close(fig)
print("Saved: residual_distribution.png")

# Residual vs Predicted
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_pred, residuals, alpha=0.3, s=10, color='steelblue')
ax.axhline(0, color='red', linestyle='--')
ax.set_title('Residual vs Nilai Prediksi', fontsize=14)
ax.set_xlabel('Harga Prediksi (£)')
ax.set_ylabel('Residual')
plt.tight_layout()
fig.savefig(f'{PLOTS_DIR}/residual_vs_predicted.png', dpi=150)
plt.close(fig)
print("Saved: residual_vs_predicted.png")

# ============================================================
# 6. SIMPAN MODEL & ARTEFAK
# ============================================================
print("\n" + "=" * 70)
print("6. MENYIMPAN MODEL & ARTEFAK")
print("=" * 70)

joblib.dump(model, f'{MODEL_DIR}/linear_regression_model.pkl')
joblib.dump(preprocessor, f'{MODEL_DIR}/preprocessor.pkl')
joblib.dump(feature_names, f'{MODEL_DIR}/feature_names.pkl')
joblib.dump(metrics, f'{MODEL_DIR}/model_metrics.pkl')
joblib.dump(le_model, f'{MODEL_DIR}/label_encoders.pkl')

print("Tersimpan:")
print(f"  {MODEL_DIR}/linear_regression_model.pkl")
print(f"  {MODEL_DIR}/preprocessor.pkl")
print(f"  {MODEL_DIR}/feature_names.pkl")
print(f"  {MODEL_DIR}/model_metrics.pkl")
print(f"  {MODEL_DIR}/label_encoders.pkl")

print("\n" + "=" * 70)
print("ANALISIS SELESAI!")
print("=" * 70)
