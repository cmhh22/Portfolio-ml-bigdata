"""
Utility Functions Module

Helper functions for visualization, data analysis, and general utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Tuple, Dict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set visualization defaults
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_distribution(data: pd.Series, title: str, 
                     xlabel: str = None, bins: int = 50,
                     save_path: Optional[Path] = None):
    """
    Plot distribution with histogram and KDE.
    
    Args:
        data: Series to plot
        title: Plot title
        xlabel: X-axis label
        bins: Number of bins
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(data, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    ax2 = ax.twinx()
    data.plot(kind='kde', ax=ax2, color='red', linewidth=2)
    
    ax.set_xlabel(xlabel or data.name)
    ax.set_ylabel('Frequency')
    ax2.set_ylabel('Density', color='red')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple = (14, 10),
                           save_path: Optional[Path] = None):
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame
        figsize: Figure size
        save_path: Path to save figure
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation
    corr = numeric_df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', 
               center=0, square=True, linewidths=0.5, 
               cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20,
                           title: str = 'Feature Importance',
                           save_path: Optional[Path] = None):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure
    """
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.show()


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray,
                            title: str = 'Actual vs Predicted',
                            save_path: Optional[Path] = None):
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Duration (seconds)')
    ax1.set_ylabel('Predicted Duration (seconds)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Residuals
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Duration (seconds)')
    ax2.set_ylabel('Residuals (seconds)')
    ax2.set_title('Residual Plot')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.show()


def plot_model_comparison(comparison_df: pd.DataFrame, 
                         metric: str = 'cv_rmse_mean',
                         save_path: Optional[Path] = None):
    """
    Plot model comparison.
    
    Args:
        comparison_df: DataFrame with model comparison results
        metric: Metric to plot
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = comparison_df['model']
    values = comparison_df[metric]
    
    bars = ax.bar(range(len(models)), values, color='steelblue', edgecolor='black')
    
    # Color the best model
    best_idx = values.idxmin() if 'rmse' in metric or 'mae' in metric else values.idxmax()
    bars[best_idx].set_color('green')
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2f}',
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.show()


def plot_temporal_patterns(df: pd.DataFrame, target_col: str = 'trip_duration',
                          save_path: Optional[Path] = None):
    """
    Plot temporal patterns (hourly, daily, weekly).
    
    Args:
        df: DataFrame with temporal columns
        target_col: Target column to analyze
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Hourly pattern
    hourly_avg = df.groupby('pickup_hour')[target_col].mean()
    axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel(f'Avg {target_col}')
    axes[0, 0].set_title('Average Trip Duration by Hour')
    axes[0, 0].grid(alpha=0.3)
    
    # Day of week pattern
    weekday_avg = df.groupby('pickup_weekday')[target_col].mean()
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[0, 1].bar(range(7), weekday_avg.values, color='steelblue', edgecolor='black')
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(weekday_names)
    axes[0, 1].set_ylabel(f'Avg {target_col}')
    axes[0, 1].set_title('Average Trip Duration by Day of Week')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Rush hour comparison
    rush_comparison = df.groupby('is_rush_hour')[target_col].mean()
    axes[1, 0].bar(['Non-Rush', 'Rush Hour'], rush_comparison.values, 
                   color=['steelblue', 'coral'], edgecolor='black')
    axes[1, 0].set_ylabel(f'Avg {target_col}')
    axes[1, 0].set_title('Rush Hour vs Non-Rush Hour')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Weekend comparison
    weekend_comparison = df.groupby('is_weekend')[target_col].mean()
    axes[1, 1].bar(['Weekday', 'Weekend'], weekend_comparison.values,
                   color=['steelblue', 'green'], edgecolor='black')
    axes[1, 1].set_ylabel(f'Avg {target_col}')
    axes[1, 1].set_title('Weekday vs Weekend')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")
    
    plt.show()


def create_interactive_map(df: pd.DataFrame, sample_size: int = 5000,
                          save_path: Optional[Path] = None):
    """
    Create interactive map of pickup/dropoff locations using Plotly.
    
    Args:
        df: DataFrame with coordinate columns
        sample_size: Number of points to plot
        save_path: Path to save HTML file
    """
    # Sample data for performance
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    # Create figure with pickup and dropoff points
    fig = go.Figure()
    
    # Pickup points
    fig.add_trace(go.Scattermapbox(
        lat=df_sample['pickup_latitude'],
        lon=df_sample['pickup_longitude'],
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.6),
        name='Pickup',
        text=df_sample.index
    ))
    
    # Dropoff points
    fig.add_trace(go.Scattermapbox(
        lat=df_sample['dropoff_latitude'],
        lon=df_sample['dropoff_longitude'],
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.6),
        name='Dropoff',
        text=df_sample.index
    ))
    
    # Layout
    fig.update_layout(
        title='NYC Taxi Pickup and Dropoff Locations',
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=40.7580, lon=-73.9855),
            zoom=10
        ),
        showlegend=True,
        height=600
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Interactive map saved to {save_path}")
    
    fig.show()


def generate_summary_statistics(df: pd.DataFrame, 
                               target_col: str = 'trip_duration') -> pd.DataFrame:
    """
    Generate comprehensive summary statistics.
    
    Args:
        df: DataFrame
        target_col: Target column
        
    Returns:
        DataFrame with summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = df[numeric_cols].describe().T
    summary['missing'] = df[numeric_cols].isnull().sum()
    summary['missing_pct'] = (summary['missing'] / len(df) * 100).round(2)
    summary['unique'] = df[numeric_cols].nunique()
    summary['skewness'] = df[numeric_cols].skew()
    summary['kurtosis'] = df[numeric_cols].kurtosis()
    
    return summary


def print_memory_usage(df: pd.DataFrame):
    """
    Print memory usage of DataFrame.
    
    Args:
        df: DataFrame
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum() / 1024**2
    
    logger.info(f"DataFrame Memory Usage:")
    logger.info(f"  Total: {total_memory:.2f} MB")
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Per row: {total_memory / len(df) * 1024:.2f} KB")
    
    # Top memory consumers
    top_cols = memory_usage.sort_values(ascending=False).head(10) / 1024**2
    logger.info("\nTop 10 memory consumers:")
    for col, mem in top_cols.items():
        if col != 'Index':
            logger.info(f"  {col}: {mem:.2f} MB")


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> Tuple[pd.Series, Dict]:
    """
    Detect outliers using IQR method.
    
    Args:
        data: Series to check
        multiplier: IQR multiplier (1.5 = mild, 3 = extreme)
        
    Returns:
        Tuple of (outlier_mask, statistics)
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    
    stats = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'n_outliers': outlier_mask.sum(),
        'outlier_pct': outlier_mask.sum() / len(data) * 100
    }
    
    return outlier_mask, stats


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def log_dataset_info(df: pd.DataFrame, name: str = "Dataset"):
    """
    Log comprehensive dataset information.
    
    Args:
        df: DataFrame
        name: Dataset name
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Information")
    logger.info(f"{'='*60}")
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"\nColumn Types:")
    logger.info(df.dtypes.value_counts().to_string())
    logger.info(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        })
        logger.info(missing_df.to_string())
    else:
        logger.info("No missing values")
    logger.info(f"{'='*60}\n")
