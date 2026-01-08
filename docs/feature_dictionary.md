# 📖 Feature Dictionary

## NYC Taxi Trip Duration - Big Data ML Project

This document describes all **26 features** used in the final prediction model.

---

## 📊 Feature Summary

| Category | Count | Features |
|----------|-------|----------|
| Original | 7 | vendor_id, passenger_count, store_and_fwd_flag, coordinates |
| Temporal | 9 | hour, dayofweek, month, binary flags, cyclic encodings |
| Geospatial | 6 | distances, direction, center point |
| Airport | 4 | distances to airports, airport trip flags |
| **TOTAL** | **26** | |

---

## 📋 Complete Feature List

The model uses exactly these 26 features in the following order:

```
1.  vendor_id            14. hour_sin
2.  passenger_count      15. hour_cos
3.  store_and_fwd_flag   16. dow_sin
4.  pickup_longitude     17. dow_cos
5.  pickup_latitude      18. haversine_distance
6.  dropoff_longitude    19. manhattan_distance
7.  dropoff_latitude     20. direction
8.  hour                 21. center_latitude
9.  dayofweek            22. center_longitude
10. month                23. dist_to_jfk_pickup
11. is_weekend           24. dist_to_lga_pickup
12. is_rush_hour         25. is_jfk_trip
13. is_night             26. is_lga_trip
```

---

## 🔢 Original Variables (7 Features)

### Categorical Variables

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `vendor_id` | Int (1-2) | Taxi provider ID | 1 = Company A, 2 = Company B |
| `passenger_count` | Int (1-6) | Number of passengers | 1-6 (most common: 1) |
| `store_and_fwd_flag` | Int (0-1) | Data stored locally before sending | 0 = No, 1 = Yes |

### Geographic Coordinates

| Variable | Type | Description | Range |
|----------|------|-------------|-------|
| `pickup_longitude` | Float | Pickup longitude | [-74.3, -73.7] |
| `pickup_latitude` | Float | Pickup latitude | [40.5, 41.0] |
| `dropoff_longitude` | Float | Dropoff longitude | [-74.3, -73.7] |
| `dropoff_latitude` | Float | Dropoff latitude | [40.5, 41.0] |

---

## 🕐 Temporal Features (9 Features)

### Component Extraction

| Variable | Type | Description | Range |
|----------|------|-------------|-------|
| `hour` | Int | Pickup hour of day | 0-23 |
| `dayofweek` | Int | Day of week (0=Monday) | 0-6 |
| `month` | Int | Month of year | 1-12 |

### Binary Temporal Indicators

| Variable | Type | Description | Criteria |
|----------|------|-------------|----------|
| `is_weekend` | Int (0-1) | Weekend indicator | dayofweek ∈ {5, 6} |
| `is_rush_hour` | Int (0-1) | Rush hour indicator | hour ∈ {7, 8, 9, 17, 18, 19} |
| `is_night` | Int (0-1) | Night time indicator | hour ∈ {22, 23, 0, 1, 2, 3, 4, 5} |

### Cyclic Encoding

| Variable | Type | Description | Formula | Range |
|----------|------|-------------|---------|-------|
| `hour_sin` | Float | Hour sine component | sin(2π × hour / 24) | [-1, 1] |
| `hour_cos` | Float | Hour cosine component | cos(2π × hour / 24) | [-1, 1] |
| `dow_sin` | Float | Day of week sine | sin(2π × dayofweek / 7) | [-1, 1] |
| `dow_cos` | Float | Day of week cosine | cos(2π × dayofweek / 7) | [-1, 1] |

**Why Cyclic Encoding?**  
Preserves the circular nature of time. Example: 23:00 is close to 00:00, which linear encoding misses.

---

## 🌍 Geospatial Features (6 Features)

### Distance Metrics

| Variable | Type | Description | Unit |
|----------|------|-------------|------|
| `haversine_distance` | Float | Great-circle distance | km |
| `manhattan_distance` | Float | Grid-based distance | km |

**Haversine Formula:**
```
a = sin²(Δφ/2) + cos(φ1)·cos(φ2)·sin²(Δλ/2)
c = 2·atan2(√a, √(1-a))
distance = R × c  (where R = 6371 km)
```

**Manhattan Formula:**
```
manhattan = |Δlat| × 111 + |Δlon| × 85  (approximate km conversion)
```

### Direction & Center Point

| Variable | Type | Description | Range/Unit |
|----------|------|-------------|------------|
| `direction` | Float | Bearing angle from pickup to dropoff | 0-360° |
| `center_latitude` | Float | Midpoint latitude | Degrees |
| `center_longitude` | Float | Midpoint longitude | Degrees |

**Direction Formula:**
```
θ = atan2(sin(Δλ)·cos(φ2), cos(φ1)·sin(φ2) - sin(φ1)·cos(φ2)·cos(Δλ))
direction = (θ × 180/π + 360) mod 360
```

---

## ✈️ Airport Features (4 Features)

### Reference Airport Locations

| Airport | Latitude | Longitude |
|---------|----------|-----------|
| JFK International | 40.6413 | -73.7781 |
| LaGuardia (LGA) | 40.7769 | -73.8740 |

### Distance to Airports

| Variable | Type | Description | Unit |
|----------|------|-------------|------|
| `dist_to_jfk_pickup` | Float | Distance from pickup to JFK | km |
| `dist_to_lga_pickup` | Float | Distance from pickup to LaGuardia | km |

### Airport Trip Indicators

| Variable | Type | Description | Criteria |
|----------|------|-------------|----------|
| `is_jfk_trip` | Int (0-1) | Trip involves JFK | pickup or dropoff within 2 km of JFK |
| `is_lga_trip` | Int (0-1) | Trip involves LaGuardia | pickup or dropoff within 2 km of LGA |

---

## 🎯 Target Variable

| Variable | Type | Description | Transformation |
|----------|------|-------------|----------------|
| `log_trip_duration` | Float | Log-transformed trip duration | log(trip_duration_seconds + 1) |

**Why Log Transform?**  
- Original `trip_duration` is highly right-skewed
- Log transformation normalizes the distribution
- Reduces impact of extreme outliers
- Improves model performance (R² improved by ~5%)

**Inverse Transform for Predictions:**
```python
trip_duration_seconds = exp(log_trip_duration) - 1
trip_duration_minutes = trip_duration_seconds / 60
```

---

## 📈 Feature Importance (Top 10)

Based on XGBoost model analysis:

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `haversine_distance` | 0.352 | Geospatial |
| 2 | `manhattan_distance` | 0.183 | Geospatial |
| 3 | `hour` | 0.098 | Temporal |
| 4 | `dropoff_latitude` | 0.067 | Original |
| 5 | `pickup_latitude` | 0.058 | Original |
| 6 | `direction` | 0.048 | Geospatial |
| 7 | `center_latitude` | 0.042 | Geospatial |
| 8 | `is_rush_hour` | 0.038 | Temporal |
| 9 | `dropoff_longitude` | 0.031 | Original |
| 10 | `is_weekend` | 0.024 | Temporal |

**Key Insights:**
- Distance features account for **>50%** of importance
- Temporal features contribute **~16%**
- Airport features have lower individual importance but improve predictions for specific routes

---

## 📊 Feature Statistics (Training Data)

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| haversine_distance | 3.44 | 4.21 | 0.0 | 32.5 |
| manhattan_distance | 4.12 | 4.89 | 0.0 | 38.2 |
| hour | 13.5 | 6.4 | 0 | 23 |
| passenger_count | 1.66 | 1.31 | 1 | 6 |
| is_weekend | 0.28 | 0.45 | 0 | 1 |
| is_rush_hour | 0.25 | 0.43 | 0 | 1 |
| is_night | 0.21 | 0.41 | 0 | 1 |

---

## 🔧 Implementation Notes

### Data Types (Optimized for Performance)

| Feature Type | Python Type | Memory Optimization |
|--------------|-------------|---------------------|
| Coordinates | float32 | 50% reduction from float64 |
| Binary flags | int8 | 87.5% reduction from int64 |
| Hour/Day/Month | int8 | 87.5% reduction from int64 |
| Distances | float32 | 50% reduction from float64 |

### Missing Value Handling

- **Original variables**: No missing values in cleaned dataset
- **Derived variables**: Calculated from existing data (no nulls)
- **Store_and_fwd_flag**: Converted from Y/N to 1/0

### Feature Generation Code Reference

See [src/feature_engineering.py](../src/feature_engineering.py) for implementation details.

---

## 📝 Features Explored but Not Included

The following features were tested during development but excluded from the final model:

| Feature | Reason for Exclusion |
|---------|---------------------|
| `pickup_cluster` | Marginal improvement, added complexity |
| `euclidean_distance` | Redundant with haversine |
| `month_sin/cos` | Low variance (data from 6 months only) |
| `bearing_sin/cos` | Direction alone sufficient |
| `dist_to_newark` | Very few trips, low impact |

---

*Last Updated: Based on model training with 1,154,992 samples*
