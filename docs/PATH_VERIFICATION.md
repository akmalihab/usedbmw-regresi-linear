# Path Verification Summary

## Struktur Path Final

### Root Directory Files
- `app.py` → mengakses `data/bmw.csv`, `model/*.pkl`

### src/train.py
- Dataset: `../data/bmw.csv`
- Output plots: `../plots/*.png`
- Output model: `../model/*.pkl`

### notebooks/analisis.ipynb
- Dataset: `../data/bmw.csv`
- Output plots: `../plots/*.png`
- Output model: `../model/*.pkl`

## Verifikasi

### Dari Root Directory
```bash
streamlit run app.py
# ✓ Akses: data/bmw.csv, model/, plots/
```

### Dari src/
```bash
cd src
python train.py
# ✓ Akses: ../data/bmw.csv
# ✓ Output: ../plots/, ../model/
```

### Dari notebooks/
```bash
cd notebooks
jupyter notebook analisis.ipynb
# ✓ Akses: ../data/bmw.csv
# ✓ Output: ../plots/, ../model/
```

## Status: ✓ Semua path sudah sesuai struktur folder

Last verified: 2026-11-05
