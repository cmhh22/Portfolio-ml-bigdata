# 🚕 NYC Taxi Trip Duration - Big Data ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Dask](https://img.shields.io/badge/Dask-Big%20Data-FDA428?style=flat)](https://dask.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **End-to-end Big Data ML project: Predicting NYC taxi trip duration with 1.5M+ records using distributed processing and advanced ML techniques**

---

## 🎯 Project Overview

A complete **Big Data Machine Learning pipeline** for predicting taxi trip duration in New York City:

- 📊 **Dataset**: 1.5M+ taxi trip records (~200MB)
- 🔧 **Processing**: Distributed computing with Dask
- 🤖 **Models**: Multiple ML algorithms (Linear, RF, XGBoost, LightGBM)
- 📈 **Analysis**: Comprehensive EDA, feature engineering, and model interpretability

### 🎓 Academic Context
This project is designed for a Big Data & Machine Learning course, demonstrating large-scale data handling, complete ML lifecycle, and professional documentation.

---

## 📁 Project Structure

```
portfolio-ml-bigdata/
├── data/
│   ├── raw/           # Original Kaggle data (train.csv, test.csv)
│   ├── processed/     # Generated: train_processed.parquet, val_processed.parquet
│   └── sampled/       # Sample submission
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA
│   ├── 02_feature_engineering.ipynb  # Feature creation
│   ├── 03_modeling.ipynb             # Model training
│   └── 04_model_interpretation.ipynb # SHAP analysis
├── app/
│   └── streamlit_app.py              # Interactive web demo
├── src/               # Python modules
├── models/            # Saved models (.pkl files)
├── visualizations/    # Generated plots
├── docs/              # Technical documentation
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/your-username/portfolio-ml-bigdata.git
cd portfolio-ml-bigdata
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Download data from Kaggle
# https://www.kaggle.com/c/nyc-taxi-trip-duration/data
# Place train.csv and test.csv in data/raw/

# 3. Run notebooks in order
jupyter notebook notebooks/
```

See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed setup instructions.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Technical Report](docs/technical_report.md) | Complete methodology and analysis |
| [Feature Dictionary](docs/feature_dictionary.md) | All features explained |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Setup and execution guide |

---

## For Academic Report

The technical documentation is in `docs/technical_report.md`:

| Requirement | Location |
|-------------|----------|
| **3.1 Dataset Description** | `docs/technical_report.md` Section 1 |
| **3.3 EDA** | `docs/technical_report.md` Section 2 + Notebook 01 |
| **3.4 Preprocessing** | `docs/technical_report.md` Section 3 |
| **3.5 Feature Engineering** | `docs/technical_report.md` Section 4 + `docs/feature_dictionary.md` |
| **3.6 Modeling** | `docs/technical_report.md` Sections 5-6 + Notebook 03 |
| **3.7 Visualizations** | `visualizations/` folder + notebook outputs |
| **3.8 Conclusions** | `docs/technical_report.md` Section 9 |

---

## 🛠️ Technologies

- **Core**: Python 3.9+, scikit-learn, XGBoost, LightGBM
- **Big Data**: Dask, PyArrow/Parquet
- **Visualization**: Matplotlib, Seaborn, Plotly, Folium
- **Analysis**: SHAP, Optuna

---

## 📊 Results

| Model | RMSE | MAE | R² | Training Time |
|-------|------|-----|-----|---------------|
| **XGBoost** | **0.3053** | **0.2199** | **0.8234** | 56s |
| LightGBM | 0.3216 | 0.2353 | 0.8040 | 27s |
| Random Forest | 0.3299 | 0.2420 | 0.7938 | 23min |
| Gradient Boosting | 0.3309 | 0.2418 | 0.7926 | 76s |
| Ridge (Baseline) | 0.4932 | 0.3799 | 0.5392 | 8s |

**Best Model:** XGBoost with R² = 0.8234 (explains 82% of trip duration variance)

**Key Finding:** `haversine_distance` alone accounts for 78% of feature importance

### 🚀 Deployment Note

> **Only LightGBM model is included in the repository** for deployment purposes.
> 
> The Streamlit demo uses LightGBM because:
> - ✅ Best balance of performance and model size
> - ✅ Fast inference time (ideal for web applications)
> - ✅ R² = 0.8040 (competitive accuracy)
> 
> **After cloning the repository:**
> 1. Run notebooks 01-03 to train all models locally
> 2. All `.pkl` files will be generated in `models/`
> 3. Local deployment will work with all trained models

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🔗 Links

- **Repository**: [GitHub - NYC Taxi Big Data ML](https://github.com/cmhh22/portfolio-ml-bigdata)
- **Dataset**: [Kaggle NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration)
- **Author**: [Carlos Manuel Hernández](https://github.com/cmhh22)
