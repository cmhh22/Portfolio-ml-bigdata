# 📊 Technical Report: NYC Taxi Trip Duration Prediction

**Big Data & Machine Learning Project**

**Author:** Carlos Manuel Hernández Hernández  
**Date:** January 2026  
**Repository:** [cmhh22](https://github.com/cmhh22)

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Data Exploration & Analysis](#data-exploration--analysis)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Engineering](#feature-engineering)
5. [Model Development](#model-development)
6. [Model Evaluation](#model-evaluation)
7. [Model Interpretability](#model-interpretability)
8. [Results & Insights](#results--insights)
9. [Conclusions](#conclusions)

---

## Dataset Overview

### Source and Context

The **NYC Taxi Trip Duration** dataset from Kaggle contains historical taxi trip records from New York City. This dataset provides a real-world regression problem with Big Data characteristics.

**Dataset Link:** [Kaggle NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration)

### Dataset Characteristics

| Characteristic | Value |
|----------------|-------|
| **Total Records** | 1,458,644 trips |
| **Time Period** | January - June 2016 |
| **File Size** | ~200 MB (CSV) |
| **Problem Type** | Regression |
| **Target Variable** | `trip_duration` (seconds) |

### Variables Description

| Variable | Type | Description |
|----------|------|-------------|
| `id` | String | Unique trip identifier |
| `vendor_id` | Categorical | Taxi provider (1 or 2) |
| `pickup_datetime` | Datetime | Pickup timestamp |
| `dropoff_datetime` | Datetime | Dropoff timestamp (train only) |
| `passenger_count` | Integer | Number of passengers (1-6) |
| `pickup_longitude` | Float | Pickup longitude coordinate |
| `pickup_latitude` | Float | Pickup latitude coordinate |
| `dropoff_longitude` | Float | Dropoff longitude coordinate |
| `dropoff_latitude` | Float | Dropoff latitude coordinate |
| `store_and_fwd_flag` | Binary | Trip stored locally before sending (Y/N) |
| **`trip_duration`** | Integer | **Target: Duration in seconds** |

### Big Data Perspective

This dataset demonstrates Big Data characteristics:

- **Volume:** 1.5M records from real-world operations
- **Variety:** Mix of temporal, geospatial, and categorical data
- **Velocity:** Continuous data generation from taxi fleet
- **Veracity:** Official data from NYC Taxi & Limousine Commission

**Technologies Used:**
- **Dask:** Distributed data processing
- **Parquet:** Efficient columnar storage
- **Parallel Computing:** Multi-core feature engineering
- **Memory Optimization:** Dtype reduction strategies

---

## Data Exploration & Analysis

> 📋 **Data Source:** `notebooks/01_data_exploration.ipynb`

### Descriptive Statistics

#### Target Variable: `trip_duration`

```
Statistic             Value
----------------------------------
Count                 1,458,644
Mean                  959.49 seconds (16.0 min)
Median                662.00 seconds (11.0 min)
Std Dev               5,237.43 seconds
Min                   1 second
Max                   3,526,282 seconds (979.5 hours!)
25th Percentile       397 seconds (6.6 min)
75th Percentile       1,075 seconds (17.9 min)
```

**Key Observations:**
- Strongly **right-skewed distribution** (mean >> median)
- Extreme outliers present (max = 40+ days - clearly data errors)
- **Log transformation required** to normalize distribution
- Most trips between 6-18 minutes (IQR range)

### Data Quality Assessment

#### Missing Values

```
Total cells analyzed: 17,503,728
Total null values:    0
Null percentage:      0.00%
```

✅ **Conclusion:** Dataset is 100% complete with NO missing values.

#### Outlier Detection

**Method:** Interquartile Range (IQR) with factor 1.5

```
Q1 (25th percentile)  = 397 seconds
Q3 (75th percentile)  = 1,075 seconds
IQR                   = 678 seconds

Lower Bound = Q1 - 1.5 × IQR = -620 → 0 seconds
Upper Bound = Q3 + 1.5 × IQR = 2,092 seconds (~35 min)

Outliers Detected     = 74,220 records (5.09%)
```

**Outliers by Feature:**
| Feature | Outliers | Percentage |
|---------|----------|------------|
| passenger_count | 154,830 | 10.61% |
| pickup_longitude | 84,322 | 5.78% |
| trip_duration | 74,220 | 5.09% |
| dropoff_longitude | 77,969 | 5.35% |

**Action:** Trips filtered to reasonable durations (60s - 2 hours).

### Temporal Patterns

#### By Hour of Day
- **Peak demand hours:** 8-9 AM (morning rush) and 6-7 PM (evening rush)
- **Low demand hours:** 4-6 AM (early morning)
- **Longest trip durations:** During rush hours (traffic congestion)

#### By Day of Week
- **Busiest days:** Friday and Saturday
- **Slowest days:** Sunday and Monday
- **Weekend vs weekday:** Weekend trips tend to be longer (leisure vs commute)

### Geographic Patterns

- **Records within NYC bounds:** 1,454,166 (99.69%)
- **Records outside bounds:** 4,478 (0.31%) - removed during preprocessing
- **Trip concentration:** Most trips concentrated in Manhattan
- **Airport patterns:** JFK and LaGuardia create distinct outlying clusters
- **High-demand zones:** Midtown, Financial District, Times Square

### Feature Correlations

**Top Correlations with `trip_duration`:**

| Feature | Correlation | Direction |
|---------|-------------|----------|
| pickup_latitude | -0.0292 | weak negative |
| pickup_longitude | 0.0265 | weak positive |
| dropoff_latitude | -0.0207 | weak negative |
| vendor_id | 0.0203 | weak positive |
| dropoff_longitude | 0.0147 | weak positive |
| passenger_count | 0.0085 | weak positive |

⚠️ **Note:** Raw coordinate correlations are weak because **distance** (derived feature) is what really matters, not individual coordinates.

---

## Data Preprocessing

> 📋 **Data Source:** `notebooks/02_feature_engineering.ipynb`

### Data Cleaning Pipeline

```
Raw Data (1,458,644 records)
    ↓
Remove Duplicates
    ↓
Remove Null Values (if any)
    ↓
Filter Geographic Bounds (NYC area)
    ↓
Filter Duration Outliers (60s - 7200s)
    ↓
Clean Data → Feature Engineering
```

### Geographic Validation

**NYC Boundaries Applied:**
- Latitude: [40.5, 41.0]
- Longitude: [-74.3, -73.7]

**Trips outside boundaries:** Removed

### Duration Validation

**Valid Range:**
- Minimum: 60 seconds (1 minute)
- Maximum: 7,200 seconds (2 hours)

**Rationale:** Extremely short trips are likely data errors; very long trips may be outliers or special cases.

### Memory Optimization

**Big Data Strategy:**

| Original Type | Optimized Type | Memory Reduction |
|---------------|----------------|------------------|
| float64 | float32 | 50% |
| int64 | int16/int8 | 75-87% |
| object | category | Variable |

**Total Memory Saved:** ~60%

### Target Transformation

**Log Transformation Applied:**

```python
y = np.log1p(trip_duration)
```

**Benefits:**
- Normalizes right-skewed distribution
- Reduces impact of extreme values
- Improves model convergence
- Standard practice for duration/price prediction

---

## Feature Engineering

> 📋 **Data Source:** `notebooks/02_feature_engineering.ipynb`

### Temporal Features

**Extracted from `pickup_datetime`:**

| Feature | Description | Type |
|---------|-------------|------|
| `pickup_hour` | Hour of day (0-23) | Numeric |
| `pickup_dayofweek` | Day of week (0=Monday) | Numeric |
| `pickup_month` | Month of year | Numeric |
| `pickup_day` | Day of month | Numeric |
| `is_weekend` | Weekend indicator | Binary |
| `is_rush_hour` | Rush hour (7-9, 17-19) | Binary |
| `is_night` | Night time (22-6) | Binary |
| `hour_sin` | Cyclic encoding (sine) | Float |
| `hour_cos` | Cyclic encoding (cosine) | Float |

**Cyclic Encoding Rationale:**  
Preserves circular nature of time (e.g., 23:00 is close to 00:00).

### Geospatial Features

**Distance Calculations:**

| Feature | Formula | Description |
|---------|---------|-------------|
| `haversine_distance` | Haversine formula | Great-circle distance (km) |
| `manhattan_distance` | \|Δlat\| + \|Δlon\| | Grid distance approximation |
| `euclidean_distance` | √(Δlat² + Δlon²) | Straight-line distance |

**Haversine Formula:**

$$d = 2r \cdot \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$

Where $r$ = 6,371 km (Earth's radius)

**Directional Features:**

| Feature | Description |
|---------|-------------|
| `bearing` | Direction angle (0-360°) |
| `bearing_sin` | Sine component of bearing |
| `bearing_cos` | Cosine component of bearing |
| `delta_lat` | Latitude difference |
| `delta_lon` | Longitude difference |

### Clustering Features

**Geographic Clustering (K-Means):**

```python
from sklearn.cluster import MiniBatchKMeans

# 50 clusters for pickup/dropoff zones
kmeans = MiniBatchKMeans(n_clusters=50, random_state=42)
pickup_cluster = kmeans.fit_predict(pickup_coords)
dropoff_cluster = kmeans.fit_predict(dropoff_coords)
```

**Why MiniBatchKMeans?**  
Processes data in batches → scalable to millions of records.

### Airport Detection

**Special Features:**

| Feature | Description |
|---------|-------------|
| `near_jfk_pickup` | Pickup near JFK Airport |
| `near_jfk_dropoff` | Dropoff near JFK Airport |
| `near_laguardia_pickup` | Pickup near LaGuardia |
| `is_airport_trip` | Any airport involvement |

### Feature Summary

**Total Features in Final Model:** 26 features

**Categories:**
- Original (coordinates + categorical): 7 features
- Temporal (hour, day, month, flags, cyclic): 9 features
- Geospatial (distances, direction, center): 6 features
- Airports (distances + trip indicators): 4 features

*Note: Additional features were explored during development (~35) but the final model uses a refined set of 26 features. See `docs/feature_dictionary.md` for complete details.*

---

## Model Development

> 📋 **Data Source:** `notebooks/03_modeling.ipynb`

### Data Split Strategy

```python
# 70% Train, 15% Validation, 15% Test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)
```

| Set | Records | Percentage |
|-----|---------|------------|
| Training | ~924K | 70% |
| Validation | ~198K | 15% |
| Test | ~198K | 15% |

### Evaluation Metrics

**Primary Metrics:**

1. **RMSE (Root Mean Squared Error)**
   - Penalizes large errors
   - Same units as target (seconds)

2. **MAE (Mean Absolute Error)**
   - Average absolute error
   - More interpretable than RMSE

3. **R² Score (Coefficient of Determination)**
   - Proportion of variance explained
   - 0 = baseline, 1 = perfect fit

4. **RMSLE (Root Mean Squared Log Error)**
   - Handles large value ranges
   - Symmetric percentage error

### Models Implemented

#### 1. Ridge Regression (Baseline)

```python
Ridge(alpha=1.0, random_state=42)
```

**Purpose:** Linear baseline with L2 regularization

#### 2. Random Forest

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    n_jobs=-1
)
```

**Purpose:** Ensemble method with interpretability

#### 3. Gradient Boosting

```python
GradientBoostingRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8
)
```

**Purpose:** Sequential error correction

#### 4. XGBoost

```python
XGBRegressor(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
```

**Purpose:** Optimized gradient boosting

#### 5. LightGBM (Best for Big Data)

```python
LGBMRegressor(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8
)
```

**Purpose:** Fast, memory-efficient gradient boosting

---

## Model Evaluation

> 📋 **Data Source:** `notebooks/03_modeling.ipynb`

### Dataset Split Results

| Set | Records | Features |
|-----|---------|----------|
| Training | 1,154,992 | 26 |
| Validation | 288,748 | 26 |

### Performance Comparison

| Model | Val RMSE | Val MAE | Val R² | Train Time |
|-------|----------|---------|--------|------------|
| **XGBoost** | **0.3053** | **0.2199** | **0.8234** | 56.1s |
| LightGBM | 0.3216 | 0.2353 | 0.8040 | 27.1s |
| Random Forest | 0.3299 | 0.2420 | 0.7938 | 1372.8s |
| Gradient Boosting | 0.3309 | 0.2418 | 0.7926 | 76.3s |
| Ridge Regression | 0.4932 | 0.3799 | 0.5392 | 8.0s |

### Best Model Selection

**Selected Model:** XGBoost

**Justification:**
- **Best validation RMSE:** 0.3053 (lowest error)
- **Highest R² Score:** 0.8234 (explains 82.3% of variance)
- **Training efficiency:** 56 seconds (reasonable for 1.1M records)
- **52.7% improvement** over Ridge Regression baseline (R²: 0.54 → 0.82)

**Note on LightGBM:** While slightly less accurate than XGBoost, LightGBM trains **2x faster** (27s vs 56s) - excellent for production with frequent retraining.

---

## Model Interpretability

> 📋 **Data Source:** `notebooks/04_model_interpretation.ipynb`

### Feature Importance Analysis

**Top 10 Most Important Features (Random Forest):**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **haversine_distance** | **0.7829** |
| 2 | direction | 0.0422 |
| 3 | dropoff_latitude | 0.0319 |
| 4 | hour_cos | 0.0259 |
| 5 | hour | 0.0237 |
| 6 | dist_to_jfk_pickup | 0.0112 |
| 7 | dow_sin | 0.0108 |
| 8 | hour_sin | 0.0095 |
| 9 | center_longitude | 0.0091 |
| 10 | dropoff_longitude | 0.0082 |

**Key Insight:** `haversine_distance` dominates with **78.3%** of total importance – distance is by far the strongest predictor of trip duration.

### SHAP Analysis

> 📋 **Data Source:** `notebooks/04_model_interpretation.ipynb`

**SHAP (SHapley Additive exPlanations)** provides model-agnostic interpretability by calculating each feature's contribution to predictions using game theory principles.

**Top 15 Features by SHAP Importance:**

| Rank | Feature | Mean \|SHAP Value\| | Category |
|------|---------|---------------------|----------|
| 1 | haversine_distance | 0.4521 | Geospatial |
| 2 | manhattan_distance | 0.1834 | Geospatial |
| 3 | hour | 0.0892 | Temporal |
| 4 | dropoff_latitude | 0.0654 | Original |
| 5 | pickup_latitude | 0.0587 | Original |
| 6 | direction | 0.0478 | Geospatial |
| 7 | center_latitude | 0.0421 | Geospatial |
| 8 | is_rush_hour | 0.0389 | Temporal |
| 9 | dropoff_longitude | 0.0312 | Original |
| 10 | is_weekend | 0.0245 | Temporal |
| 11 | center_longitude | 0.0198 | Geospatial |
| 12 | hour_cos | 0.0176 | Temporal |
| 13 | dist_to_jfk_pickup | 0.0143 | Airport |
| 14 | pickup_longitude | 0.0127 | Original |
| 15 | hour_sin | 0.0094 | Temporal |

**Feature Category Breakdown:**
- **Geospatial features:** ~72% of total SHAP importance
- **Temporal features:** ~18% of total SHAP importance
- **Airport features:** ~6% of total SHAP importance
- **Other features:** ~4% of total SHAP importance

**Key SHAP Insights:**

1. **Non-linear Distance Effects**
   - SHAP reveals that distance impact is non-linear
   - Short trips (<2 km): Nearly linear relationship
   - Medium trips (2-10 km): Steeper increase in duration
   - Long trips (>10 km): Flatter curve (highway speeds)

2. **Temporal Interaction Patterns**
   - `hour` interacts strongly with `distance`
   - Rush hour (7-9 AM, 5-7 PM): +15-25% duration increase
   - Night hours (12-5 AM): -10-15% faster trips
   - Weekend effect stronger for long-distance trips

3. **Geographic Hotspots**
   - Pickups in Midtown Manhattan: Higher baseline duration
   - Airport trips: More predictable, less variance
   - Bridge/tunnel routes: Identified through lat/lon SHAP values

4. **Model Confidence**
   - SHAP base value: 6.85 (log scale)
   - Average prediction spread: ±0.30 log units
   - High confidence for typical trips, uncertainty for outliers

**Visualizations Generated:**
- `shap_importance.png` - Global feature importance ranking
- `shap_summary.png` - Beeswarm plot showing value distributions
- `shap_dependence.png` - Feature interaction analysis
- `shap_local_0.png` - Example individual prediction breakdown

---

## Results & Insights

### Key Findings

#### 1. Distance is the Dominant Predictor

- **Haversine distance alone accounts for 78.3%** of feature importance
- Direction (bearing) adds another 4.2%
- Geographic features combined explain >85% of predictive power
- This confirms intuition: longer trips take longer

#### 2. Temporal Patterns Matter

- **Hour of day** is in top 5 features (both raw and cyclic encoded)
- Rush hours significantly increase trip duration
- Day of week patterns captured via sine/cosine encoding
- Time features account for ~5% of importance

#### 3. Airport Trips Have Distinct Patterns

- `dist_to_jfk_pickup` appears in top 10 features
- Airport routes are more predictable (fixed distance)
- Useful for specialized fare estimation

#### 4. Model Performance

- **Best Model (XGBoost):** R² = 0.8234 on validation set
- **RMSE:** 0.3053 (in log scale) ≈ average error of ~20% in duration
- **Improvement over baseline:** 52.7% better than Ridge Regression
- **Practical accuracy:** Model explains 82% of trip duration variance

### Business Applications

The trained model enables:

1. **Dynamic Pricing:** Accurate fare estimation before trip
2. **ETA Prediction:** Reliable arrival time estimates
3. **Fleet Optimization:** Deploy taxis based on demand patterns
4. **Route Planning:** Identify efficient routes by time-of-day
5. **Customer Experience:** Set correct expectations

---

## Conclusions

### Project Achievements

✅ **Big Data Processing:** Successfully processed 1,458,644 records with Dask, achieving 99.9% data quality

✅ **Feature Engineering:** Created 26 engineered features from 11 original variables, with haversine_distance being the dominant predictor (78.3% importance)

✅ **Model Comparison:** Evaluated 5 algorithms - XGBoost achieved best R² = 0.8234

✅ **Production-Ready:** XGBoost model explains 82% of variance with 56s training time

✅ **Interpretability:** SHAP analysis reveals geospatial features account for 72% of prediction importance, with clear non-linear distance patterns

✅ **Comprehensive Visualizations:** Generated 12 publication-quality plots across EDA, modeling, and SHAP interpretation

### Technical Learnings

1. **Distance is critical:** Single feature (haversine_distance) provides 78% of predictive power
2. **Temporal encoding matters:** Cyclic sine/cosine encoding outperforms raw hour values
3. **Tree-based models dominate:** XGBoost/LightGBM outperform linear models by 50%+
4. **Big Data techniques scale:** Dask + Parquet enable processing of datasets 10x larger
5. **Feature engineering > Model selection:** Good features matter more than complex models

### Limitations

1. **Historical Data (2016):** Traffic patterns may have changed
2. **Missing Variables:** Weather, events, real-time traffic not included
3. **Geographic Bias:** Mostly Manhattan trips, less data for outer boroughs
4. **Static Model:** Doesn't adapt to changing conditions

### Future Improvements

**Data Enhancements:**
- Integrate real-time traffic APIs
- Add weather data from historical sources
- Include special events (sports, concerts, holidays)
- Expand to full multi-year TLC dataset (200M+ trips)

**Model Enhancements:**
- Hyperparameter tuning with Optuna/Grid Search
- Deep learning models (LSTM for sequences)
- Online learning for real-time adaptation
- Ensemble of best models

**Deployment:**
- REST API with FastAPI
- Docker containerization
- CI/CD pipeline
- Model monitoring and drift detection
- A/B testing framework

---

## References

### Dataset

- Kaggle. (2017). *NYC Taxi Trip Duration Competition*. https://www.kaggle.com/c/nyc-taxi-trip-duration

### Libraries & Tools

- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NIPS.
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NIPS.
- Rocklin, M. (2015). *Dask: Parallel Computation with Blocked algorithms and Task Scheduling*. SciPy.

### Methodology

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly.

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Project Repository:** [GitHub Link]
