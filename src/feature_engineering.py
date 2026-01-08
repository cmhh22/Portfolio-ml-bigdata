"""
Feature Engineering Module

Advanced feature engineering for NYC Taxi Trip Duration prediction:
- Temporal features (hour, day, week, rush hours)
- Geospatial features (distance, speed, direction, areas)
- Interaction features
- Aggregated features
- Big Data optimized
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.filterwarnings('ignore')

from src.config import (
    NYC_CENTER, AIRPORTS, 
    MORNING_RUSH, EVENING_RUSH,
    RANDOM_STATE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering optimized for Big Data scenarios.
    """
    
    def __init__(self):
        self.pickup_clusters = None
        self.dropoff_clusters = None
        self.n_clusters = 20
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from pickup_datetime.
        
        Args:
            df: DataFrame with 'pickup_datetime' column
            
        Returns:
            DataFrame with temporal features
        """
        logger.info("Creating temporal features...")
        df = df.copy()
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df['pickup_datetime']):
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        
        # Basic time components
        df['pickup_year'] = df['pickup_datetime'].dt.year
        df['pickup_month'] = df['pickup_datetime'].dt.month
        df['pickup_day'] = df['pickup_datetime'].dt.day
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_minute'] = df['pickup_datetime'].dt.minute
        df['pickup_weekday'] = df['pickup_datetime'].dt.weekday  # 0=Monday, 6=Sunday
        df['pickup_week'] = df['pickup_datetime'].dt.isocalendar().week
        
        # Day of year
        df['pickup_dayofyear'] = df['pickup_datetime'].dt.dayofyear
        
        # Binary features
        df['is_weekend'] = (df['pickup_weekday'] >= 5).astype(int)
        df['is_holiday'] = self._is_holiday(df['pickup_datetime']).astype(int)
        
        # Rush hour features
        df['is_morning_rush'] = ((df['pickup_hour'] >= MORNING_RUSH[0]) & 
                                  (df['pickup_hour'] < MORNING_RUSH[1])).astype(int)
        df['is_evening_rush'] = ((df['pickup_hour'] >= EVENING_RUSH[0]) & 
                                  (df['pickup_hour'] < EVENING_RUSH[1])).astype(int)
        df['is_rush_hour'] = ((df['is_morning_rush'] == 1) | 
                              (df['is_evening_rush'] == 1)).astype(int)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(df['pickup_hour'], 
                                   bins=[0, 6, 12, 18, 24],
                                   labels=['night', 'morning', 'afternoon', 'evening'],
                                   include_lowest=True)
        
        # Cyclical encoding for hour (preserves circular nature)
        df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
        
        # Cyclical encoding for day of week
        df['weekday_sin'] = np.sin(2 * np.pi * df['pickup_weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['pickup_weekday'] / 7)
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['pickup_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['pickup_month'] / 12)
        
        logger.info(f"Created {20} temporal features")
        return df
    
    def _is_holiday(self, dates: pd.Series) -> pd.Series:
        """
        Mark US holidays for the year 2016.
        
        Args:
            dates: Series of datetime objects
            
        Returns:
            Boolean series indicating holidays
        """
        holidays_2016 = [
            '2016-01-01',  # New Year
            '2016-01-18',  # MLK Day
            '2016-02-15',  # Presidents Day
            '2016-05-30',  # Memorial Day
            '2016-07-04',  # Independence Day
            '2016-09-05',  # Labor Day
            '2016-10-10',  # Columbus Day
            '2016-11-11',  # Veterans Day
            '2016-11-24',  # Thanksgiving
            '2016-12-25',  # Christmas
        ]
        
        holiday_dates = pd.to_datetime(holidays_2016).date
        return dates.dt.date.isin(holiday_dates)
    
    def create_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create geography-based features.
        
        Args:
            df: DataFrame with coordinate columns
            
        Returns:
            DataFrame with geospatial features
        """
        logger.info("Creating geospatial features...")
        df = df.copy()
        
        # Haversine distance (great-circle distance)
        df['distance_haversine_km'] = self._haversine_distance(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
        
        # Manhattan distance (city blocks approximation)
        df['distance_manhattan_km'] = self._manhattan_distance(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
        
        # Direction/bearing
        df['direction'] = self._calculate_bearing(
            df['pickup_latitude'], df['pickup_longitude'],
            df['dropoff_latitude'], df['dropoff_longitude']
        )
        
        # Distance to NYC center
        df['pickup_distance_to_center'] = self._haversine_distance(
            df['pickup_latitude'], df['pickup_longitude'],
            np.full(len(df), NYC_CENTER['lat']),
            np.full(len(df), NYC_CENTER['lon'])
        )
        
        df['dropoff_distance_to_center'] = self._haversine_distance(
            df['dropoff_latitude'], df['dropoff_longitude'],
            np.full(len(df), NYC_CENTER['lat']),
            np.full(len(df), NYC_CENTER['lon'])
        )
        
        # Airport features
        for airport_name, airport_info in AIRPORTS.items():
            # Distance to airport
            df[f'pickup_distance_to_{airport_name}'] = self._haversine_distance(
                df['pickup_latitude'], df['pickup_longitude'],
                np.full(len(df), airport_info['lat']),
                np.full(len(df), airport_info['lon'])
            )
            
            df[f'dropoff_distance_to_{airport_name}'] = self._haversine_distance(
                df['dropoff_latitude'], df['dropoff_longitude'],
                np.full(len(df), airport_info['lat']),
                np.full(len(df), airport_info['lon'])
            )
            
            # Near airport binary features
            df[f'pickup_near_{airport_name}'] = (
                df[f'pickup_distance_to_{airport_name}'] < airport_info['radius_km']
            ).astype(int)
            
            df[f'dropoff_near_{airport_name}'] = (
                df[f'dropoff_distance_to_{airport_name}'] < airport_info['radius_km']
            ).astype(int)
        
        # Trip to/from airport
        df['trip_to_airport'] = (
            (df['dropoff_near_JFK'] == 1) | 
            (df['dropoff_near_LGA'] == 1) | 
            (df['dropoff_near_EWR'] == 1)
        ).astype(int)
        
        df['trip_from_airport'] = (
            (df['pickup_near_JFK'] == 1) | 
            (df['pickup_near_LGA'] == 1) | 
            (df['pickup_near_EWR'] == 1)
        ).astype(int)
        
        # Speed estimate (if trip_duration exists)
        if 'trip_duration' in df.columns:
            df['avg_speed_haversine'] = (
                df['distance_haversine_km'] / (df['trip_duration'] / 3600)
            ).replace([np.inf, -np.inf], np.nan).fillna(0)
            
            df['avg_speed_manhattan'] = (
                df['distance_manhattan_km'] / (df['trip_duration'] / 3600)
            ).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        logger.info(f"Created {len([col for col in df.columns if col.startswith(('distance_', 'direction', 'pickup_', 'dropoff_', 'trip_', 'avg_speed'))])} geospatial features")
        return df
    
    def _haversine_distance(self, lat1: pd.Series, lon1: pd.Series, 
                           lat2: pd.Series, lon2: pd.Series) -> pd.Series:
        """
        Calculate haversine distance between coordinates (vectorized).
        
        Args:
            lat1, lon1: Starting coordinates
            lat2, lon2: Ending coordinates
            
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of Earth in km
        r = 6371
        
        return c * r
    
    def _manhattan_distance(self, lat1: pd.Series, lon1: pd.Series,
                          lat2: pd.Series, lon2: pd.Series) -> pd.Series:
        """
        Calculate Manhattan distance (city blocks).
        
        Args:
            lat1, lon1: Starting coordinates
            lat2, lon2: Ending coordinates
            
        Returns:
            Approximate distance in kilometers
        """
        # Approximate: 1 degree latitude ≈ 111 km, 1 degree longitude ≈ 85 km (at NYC latitude)
        lat_diff_km = np.abs(lat2 - lat1) * 111
        lon_diff_km = np.abs(lon2 - lon1) * 85
        
        return lat_diff_km + lon_diff_km
    
    def _calculate_bearing(self, lat1: pd.Series, lon1: pd.Series,
                         lat2: pd.Series, lon2: pd.Series) -> pd.Series:
        """
        Calculate bearing/direction between two points.
        
        Args:
            lat1, lon1: Starting coordinates
            lat2, lon2: Ending coordinates
            
        Returns:
            Bearing in degrees (0-360)
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        
        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        bearing = np.arctan2(x, y)
        bearing = np.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    def create_cluster_features(self, df: pd.DataFrame, 
                               fit: bool = False) -> pd.DataFrame:
        """
        Create cluster-based features for pickup/dropoff locations.
        Uses MiniBatchKMeans for memory efficiency.
        
        Args:
            df: DataFrame with coordinate columns
            fit: Whether to fit new clusters (training) or use existing
            
        Returns:
            DataFrame with cluster features
        """
        logger.info(f"Creating cluster features (fit={fit})...")
        df = df.copy()
        
        if fit:
            # Fit pickup clusters
            pickup_coords = df[['pickup_latitude', 'pickup_longitude']].values
            self.pickup_clusters = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=RANDOM_STATE,
                batch_size=10000
            )
            df['pickup_cluster'] = self.pickup_clusters.fit_predict(pickup_coords)
            
            # Fit dropoff clusters
            dropoff_coords = df[['dropoff_latitude', 'dropoff_longitude']].values
            self.dropoff_clusters = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=RANDOM_STATE,
                batch_size=10000
            )
            df['dropoff_cluster'] = self.dropoff_clusters.fit_predict(dropoff_coords)
        else:
            if self.pickup_clusters is None or self.dropoff_clusters is None:
                raise ValueError("Clusters not fitted. Set fit=True for training data.")
            
            pickup_coords = df[['pickup_latitude', 'pickup_longitude']].values
            df['pickup_cluster'] = self.pickup_clusters.predict(pickup_coords)
            
            dropoff_coords = df[['dropoff_latitude', 'dropoff_longitude']].values
            df['dropoff_cluster'] = self.dropoff_clusters.predict(dropoff_coords)
        
        logger.info(f"Created cluster features with {self.n_clusters} clusters")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        df = df.copy()
        
        # Distance × Time interactions
        if 'distance_haversine_km' in df.columns and 'pickup_hour' in df.columns:
            df['distance_hour_interaction'] = df['distance_haversine_km'] * df['pickup_hour']
            df['distance_rush_interaction'] = df['distance_haversine_km'] * df['is_rush_hour']
            df['distance_weekend_interaction'] = df['distance_haversine_km'] * df['is_weekend']
        
        # Vendor × Other features (if vendor exists)
        if 'vendor_id' in df.columns:
            df['vendor_rush_interaction'] = df['vendor_id'] * df['is_rush_hour']
            df['vendor_weekend_interaction'] = df['vendor_id'] * df['is_weekend']
            
            if 'distance_haversine_km' in df.columns:
                df['vendor_distance_interaction'] = df['vendor_id'] * df['distance_haversine_km']
        
        # Passenger count interactions (if exists)
        if 'passenger_count' in df.columns:
            if 'distance_haversine_km' in df.columns:
                df['passenger_distance_interaction'] = df['passenger_count'] * df['distance_haversine_km']
            df['passenger_rush_interaction'] = df['passenger_count'] * df['is_rush_hour']
        
        logger.info(f"Created {len([col for col in df.columns if '_interaction' in col])} interaction features")
        return df
    
    def create_aggregated_features(self, df: pd.DataFrame, 
                                  fit: bool = False) -> pd.DataFrame:
        """
        Create aggregated features (mean duration per route, etc.).
        Only applicable when training data with trip_duration is available.
        
        Args:
            df: DataFrame
            fit: Whether to calculate aggregations (training only)
            
        Returns:
            DataFrame with aggregated features
        """
        if 'trip_duration' not in df.columns:
            logger.info("Skipping aggregated features (no trip_duration)")
            return df
        
        logger.info("Creating aggregated features...")
        df = df.copy()
        
        # Hour-based aggregations
        hour_stats = df.groupby('pickup_hour')['trip_duration'].agg([
            ('hour_mean_duration', 'mean'),
            ('hour_std_duration', 'std'),
            ('hour_median_duration', 'median')
        ]).reset_index()
        df = df.merge(hour_stats, on='pickup_hour', how='left')
        
        # Weekday aggregations
        weekday_stats = df.groupby('pickup_weekday')['trip_duration'].agg([
            ('weekday_mean_duration', 'mean'),
            ('weekday_std_duration', 'std')
        ]).reset_index()
        df = df.merge(weekday_stats, on='pickup_weekday', how='left')
        
        # Cluster-based aggregations
        if 'pickup_cluster' in df.columns and 'dropoff_cluster' in df.columns:
            route_stats = df.groupby(['pickup_cluster', 'dropoff_cluster'])['trip_duration'].agg([
                ('route_mean_duration', 'mean'),
                ('route_std_duration', 'std'),
                ('route_count', 'count')
            ]).reset_index()
            df = df.merge(route_stats, on=['pickup_cluster', 'dropoff_cluster'], how='left')
        
        logger.info(f"Created aggregated features")
        return df
    
    def engineer_all_features(self, df: pd.DataFrame, 
                             fit: bool = False) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit transformations (training=True, test=False)
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info(f"Starting complete feature engineering (fit={fit})...")
        
        # Temporal features
        df = self.create_temporal_features(df)
        
        # Geospatial features
        df = self.create_geospatial_features(df)
        
        # Cluster features
        df = self.create_cluster_features(df, fit=fit)
        
        # Interaction features
        df = self.create_interaction_features(df)
        
        # Aggregated features (only for training with target)
        if fit and 'trip_duration' in df.columns:
            df = self.create_aggregated_features(df, fit=fit)
        
        logger.info(f"Feature engineering complete: {df.shape[1]} total features")
        
        return df
