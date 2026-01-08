"""
Preprocessing Module

Scalable data preprocessing pipeline with Big Data support:
- Distributed processing with Dask
- Outlier detection and removal
- Missing value handling
- Data validation
- Memory optimization
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from typing import Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

from src.config import (
    MIN_TRIP_DURATION, MAX_TRIP_DURATION,
    MIN_DISTANCE, MAX_DISTANCE,
    MIN_SPEED, MAX_SPEED,
    RANDOM_STATE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BigDataPreprocessor:
    """
    Preprocessor with support for both Pandas and Dask DataFrames.
    Optimized for Big Data scenarios.
    """
    
    def __init__(self, scaler_type: str = 'robust'):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        self.numeric_features = []
        self.stats = {}
        
    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Validate data quality and remove invalid records.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned_df, validation_report)
        """
        logger.info("Validating data quality...")
        initial_rows = len(df)
        report = {
            'initial_rows': initial_rows,
            'removed': {},
            'final_rows': 0,
            'data_quality_score': 0.0
        }
        
        # Check for required columns
        required_cols = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                        'dropoff_longitude', 'dropoff_latitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicates
        dup_mask = df.duplicated()
        report['removed']['duplicates'] = dup_mask.sum()
        df = df[~dup_mask].copy()
        
        # Remove missing values
        missing_mask = df[required_cols].isnull().any(axis=1)
        report['removed']['missing_values'] = missing_mask.sum()
        df = df[~missing_mask].copy()
        
        # Remove invalid coordinates (NYC area: lat 40.5-41.0, lon -74.5 to -73.5)
        coord_mask = (
            (df['pickup_latitude'] < 40.5) | (df['pickup_latitude'] > 41.0) |
            (df['pickup_longitude'] < -74.5) | (df['pickup_longitude'] > -73.5) |
            (df['dropoff_latitude'] < 40.5) | (df['dropoff_latitude'] > 41.0) |
            (df['dropoff_longitude'] < -74.5) | (df['dropoff_longitude'] > -73.5)
        )
        report['removed']['invalid_coordinates'] = coord_mask.sum()
        df = df[~coord_mask].copy()
        
        # Remove invalid trip durations (if exists)
        if 'trip_duration' in df.columns:
            duration_mask = (
                (df['trip_duration'] < MIN_TRIP_DURATION) |
                (df['trip_duration'] > MAX_TRIP_DURATION)
            )
            report['removed']['invalid_duration'] = duration_mask.sum()
            df = df[~duration_mask].copy()
        
        report['final_rows'] = len(df)
        report['retention_rate'] = len(df) / initial_rows
        report['data_quality_score'] = len(df) / initial_rows * 100
        
        logger.info(f"Validation complete: {initial_rows:,} → {len(df):,} rows "
                   f"({report['retention_rate']:.2%} retained)")
        
        return df, report
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       columns: Optional[list] = None) -> Tuple[pd.DataFrame, dict]:
        """
        Remove outliers using IQR or Z-score method.
        
        Args:
            df: Input DataFrame
            method: 'iqr' or 'zscore'
            columns: Columns to check for outliers (None = auto-detect)
            
        Returns:
            Tuple of (cleaned_df, outlier_report)
        """
        logger.info(f"Removing outliers using {method} method...")
        initial_rows = len(df)
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude IDs
            columns = [col for col in columns if 'id' not in col.lower()]
        
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        outlier_counts = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                col_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            else:  # zscore
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_mask = z_scores > 3
            
            outlier_counts[col] = col_mask.sum()
            outlier_mask |= col_mask
        
        df_clean = df[~outlier_mask].copy()
        
        report = {
            'method': method,
            'initial_rows': initial_rows,
            'outliers_removed': outlier_mask.sum(),
            'final_rows': len(df_clean),
            'retention_rate': len(df_clean) / initial_rows,
            'outliers_by_column': outlier_counts
        }
        
        logger.info(f"Removed {outlier_mask.sum():,} outliers "
                   f"({outlier_mask.sum()/initial_rows:.2%})")
        
        return df_clean, report
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values with various strategies.
        
        Args:
            df: Input DataFrame
            strategy: 'mean', 'median', 'mode', 'drop'
            
        Returns:
            DataFrame with handled missing values
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        missing_report = df.isnull().sum()
        missing_cols = missing_report[missing_report > 0]
        
        if len(missing_cols) == 0:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Found missing values in {len(missing_cols)} columns")
        
        df = df.copy()
        
        if strategy == 'drop':
            df = df.dropna()
        else:
            for col in missing_cols.index:
                if df[col].dtype in ['int64', 'float64']:
                    if strategy == 'mean':
                        fill_value = df[col].mean()
                    elif strategy == 'median':
                        fill_value = df[col].median()
                    else:
                        fill_value = df[col].mode()[0]
                    
                    df[col].fillna(fill_value, inplace=True)
                else:
                    # For categorical, use mode
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        logger.info("Missing values handled successfully")
        return df
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting dtypes.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        logger.info("Optimizing data types for memory efficiency...")
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        df = df.copy()
        
        # Optimize integers
        int_cols = df.select_dtypes(include=['int']).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Optimize floats
        float_cols = df.select_dtypes(include=['float']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns with few unique values to category
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (1 - final_memory / initial_memory) * 100
        
        logger.info(f"Memory reduced: {initial_memory:.2f}MB → {final_memory:.2f}MB "
                   f"({reduction:.1f}% reduction)")
        
        return df
    
    def fit_scaler(self, df: pd.DataFrame, columns: list):
        """
        Fit scaler on specified columns.
        
        Args:
            df: Training DataFrame
            columns: Columns to scale
        """
        logger.info(f"Fitting {self.scaler_type} scaler on {len(columns)} features")
        self.numeric_features = columns
        self.scaler.fit(df[columns])
        
        # Store statistics
        self.stats = {
            'mean': df[columns].mean().to_dict(),
            'std': df[columns].std().to_dict(),
            'min': df[columns].min().to_dict(),
            'max': df[columns].max().to_dict()
        }
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted scaler to DataFrame.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.numeric_features:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        df = df.copy()
        df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame
            columns: Columns to scale
            
        Returns:
            Transformed DataFrame
        """
        self.fit_scaler(df, columns)
        return self.transform(df)
    
    def save(self, path: str):
        """Save preprocessor to disk."""
        joblib.dump({
            'scaler': self.scaler,
            'numeric_features': self.numeric_features,
            'stats': self.stats,
            'scaler_type': self.scaler_type
        }, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: str):
        """Load preprocessor from disk."""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.numeric_features = data['numeric_features']
        self.stats = data['stats']
        self.scaler_type = data['scaler_type']
        logger.info(f"Preprocessor loaded from {path}")


def preprocess_pipeline(df: pd.DataFrame, 
                       is_training: bool = True,
                       preprocessor: Optional[BigDataPreprocessor] = None,
                       validate: bool = True,
                       remove_outliers: bool = True,
                       optimize_memory: bool = True) -> Union[Tuple[pd.DataFrame, BigDataPreprocessor, dict], 
                                                               Tuple[pd.DataFrame, dict]]:
    """
    Complete preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        is_training: Whether this is training data
        preprocessor: Fitted preprocessor (for test data)
        validate: Whether to validate data
        remove_outliers: Whether to remove outliers
        optimize_memory: Whether to optimize memory
        
    Returns:
        If training: (processed_df, preprocessor, report)
        If testing: (processed_df, report)
    """
    logger.info(f"Starting preprocessing pipeline (training={is_training})")
    report = {}
    
    # Step 1: Validation
    if validate:
        df, validation_report = preprocessor.validate_data(df) if preprocessor else BigDataPreprocessor().validate_data(df)
        report['validation'] = validation_report
    
    # Step 2: Remove outliers (only on training)
    if is_training and remove_outliers:
        temp_preprocessor = BigDataPreprocessor()
        df, outlier_report = temp_preprocessor.remove_outliers(df)
        report['outliers'] = outlier_report
    
    # Step 3: Handle missing values
    df = BigDataPreprocessor().handle_missing_values(df, strategy='median')
    
    # Step 4: Optimize memory
    if optimize_memory:
        df = BigDataPreprocessor().optimize_dtypes(df)
    
    report['final_shape'] = df.shape
    report['final_memory_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    
    logger.info(f"Preprocessing complete: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    if is_training:
        if preprocessor is None:
            preprocessor = BigDataPreprocessor()
        return df, preprocessor, report
    else:
        return df, report
