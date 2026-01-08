"""
Data Loading Module

Efficient data loading with support for Big Data techniques:
- Chunked reading for large files
- Dask for distributed processing
- Stratified sampling for quick iteration
- Memory optimization
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from pathlib import Path
from typing import Optional, Union, Tuple
import logging
from tqdm import tqdm

from src.config import (
    TRAIN_FILE, TEST_FILE, TRAIN_SAMPLE, 
    CHUNK_SIZE, DASK_BLOCKSIZE, SAMPLE_SIZE, SAMPLE_FRACTION,
    TARGET_COLUMN, RANDOM_STATE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_pandas(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
    usecols: Optional[list] = None,
    dtype: Optional[dict] = None
) -> pd.DataFrame:
    """
    Load data using pandas (for small to medium datasets).
    
    Args:
        file_path: Path to CSV file
        nrows: Number of rows to load (None = all)
        usecols: Columns to load (None = all)
        dtype: Data types for columns
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(
            file_path,
            nrows=nrows,
            usecols=usecols,
            dtype=dtype,
            parse_dates=['pickup_datetime'] if 'pickup_datetime' in (usecols or []) or usecols is None else False
        )
        
        logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def load_data_dask(
    file_path: Union[str, Path],
    blocksize: str = DASK_BLOCKSIZE,
    usecols: Optional[list] = None
) -> dd.DataFrame:
    """
    Load data using Dask for large datasets (lazy loading).
    
    Args:
        file_path: Path to CSV file
        blocksize: Size of each partition (e.g., '25MB')
        usecols: Columns to load
        
    Returns:
        Dask DataFrame (lazy)
    """
    logger.info(f"Loading data with Dask from {file_path}")
    
    try:
        ddf = dd.read_csv(
            file_path,
            blocksize=blocksize,
            usecols=usecols,
            parse_dates=['pickup_datetime'],
            assume_missing=True
        )
        
        logger.info(f"Created Dask DataFrame with {ddf.npartitions} partitions")
        
        return ddf
        
    except Exception as e:
        logger.error(f"Error loading data with Dask: {str(e)}")
        raise


def load_data_chunked(
    file_path: Union[str, Path],
    chunksize: int = CHUNK_SIZE,
    process_func: Optional[callable] = None
) -> pd.DataFrame:
    """
    Load data in chunks and optionally process each chunk.
    Useful for preprocessing large files that don't fit in memory.
    
    Args:
        file_path: Path to CSV file
        chunksize: Number of rows per chunk
        process_func: Function to apply to each chunk (optional)
        
    Returns:
        DataFrame with processed data (if process_func concatenates results)
    """
    logger.info(f"Loading data in chunks from {file_path}")
    
    chunks = []
    
    try:
        # Count total rows for progress bar
        total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
        n_chunks = (total_rows // chunksize) + 1
        
        reader = pd.read_csv(
            file_path,
            chunksize=chunksize,
            parse_dates=['pickup_datetime']
        )
        
        for chunk in tqdm(reader, total=n_chunks, desc="Processing chunks"):
            if process_func:
                processed_chunk = process_func(chunk)
                chunks.append(processed_chunk)
            else:
                chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded and processed {len(df):,} rows")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data in chunks: {str(e)}")
        raise


def create_stratified_sample(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    n_samples: int = SAMPLE_SIZE,
    stratify_bins: int = 10,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Create a stratified sample that preserves the distribution of the target.
    Essential for Big Data: allows fast iteration on representative subset.
    
    Args:
        df: Full dataset
        target_col: Target column name
        n_samples: Number of samples to extract
        stratify_bins: Number of bins for stratification
        random_state: Random seed
        
    Returns:
        Stratified sample
    """
    logger.info(f"Creating stratified sample of {n_samples:,} rows")
    
    # Create bins for target (for stratification)
    df['_target_bin'] = pd.qcut(
        df[target_col], 
        q=stratify_bins, 
        labels=False, 
        duplicates='drop'
    )
    
    # Stratified sampling
    sample = df.groupby('_target_bin', group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), n_samples // stratify_bins),
            random_state=random_state
        )
    )
    
    # Remove temporary column
    sample = sample.drop('_target_bin', axis=1)
    df = df.drop('_target_bin', axis=1)
    
    logger.info(f"Sample size: {len(sample):,} rows ({len(sample)/len(df)*100:.2f}% of original)")
    logger.info(f"Sample target distribution: mean={sample[target_col].mean():.2f}, "
                f"std={sample[target_col].std():.2f}")
    logger.info(f"Original target distribution: mean={df[target_col].mean():.2f}, "
                f"std={df[target_col].std():.2f}")
    
    return sample


def create_sample(
    input_file: Path = TRAIN_FILE,
    output_file: Path = TRAIN_SAMPLE,
    n_samples: int = SAMPLE_SIZE,
    force: bool = False
) -> pd.DataFrame:
    """
    Create and save a stratified sample from the full dataset.
    
    Args:
        input_file: Path to full dataset
        output_file: Path to save sample
        n_samples: Number of samples
        force: Force recreation even if sample exists
        
    Returns:
        Sample DataFrame
    """
    if output_file.exists() and not force:
        logger.info(f"Sample already exists at {output_file}, loading...")
        return pd.read_csv(output_file, parse_dates=['pickup_datetime'])
    
    logger.info(f"Creating new sample from {input_file}")
    
    # Load full data
    df = load_data_pandas(input_file)
    
    # Create sample
    sample = create_stratified_sample(df, n_samples=n_samples)
    
    # Save sample
    sample.to_csv(output_file, index=False)
    logger.info(f"Sample saved to {output_file}")
    
    return sample


def load_train_test_split(
    train_file: Path = TRAIN_FILE,
    test_file: Path = TEST_FILE,
    use_sample: bool = False,
    sample_size: int = SAMPLE_SIZE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test datasets.
    
    Args:
        train_file: Path to training data
        test_file: Path to test data
        use_sample: Whether to use a sample (for quick iteration)
        sample_size: Size of sample if use_sample=True
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if use_sample:
        logger.info("Using sample data for quick iteration")
        if TRAIN_SAMPLE.exists():
            train = pd.read_csv(TRAIN_SAMPLE, parse_dates=['pickup_datetime'])
        else:
            train = load_data_pandas(train_file)
            train = create_stratified_sample(train, n_samples=sample_size)
            train.to_csv(TRAIN_SAMPLE, index=False)
    else:
        train = load_data_pandas(train_file)
    
    test = load_data_pandas(test_file)
    
    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Test shape: {test.shape}")
    
    return train, test


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    Important for Big Data to reduce memory footprint.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with optimized dtypes
    """
    logger.info("Optimizing data types...")
    
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Optimize integers
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Optimize floats
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert object to category if cardinality is low
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    logger.info(f"Memory reduced from {original_memory:.2f} MB to {final_memory:.2f} MB "
                f"({(1 - final_memory/original_memory)*100:.1f}% reduction)")
    
    return df


def save_processed_data(
    df: pd.DataFrame,
    file_path: Path,
    format: str = 'parquet'
):
    """
    Save processed data in efficient format.
    
    Args:
        df: DataFrame to save
        file_path: Output path
        format: Format ('parquet' or 'csv')
    """
    logger.info(f"Saving data to {file_path}")
    
    if format == 'parquet':
        df.to_parquet(file_path, compression='snappy', index=False)
    else:
        df.to_csv(file_path, index=False)
    
    logger.info(f"Data saved successfully")


def load_processed_data(file_path: Path) -> pd.DataFrame:
    """
    Load processed data (parquet or csv).
    
    Args:
        file_path: Path to processed data
        
    Returns:
        DataFrame
    """
    logger.info(f"Loading processed data from {file_path}")
    
    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    
    logger.info(f"Loaded {len(df):,} rows")
    
    return df


# Example usage
if __name__ == "__main__":
    # Create sample for quick iteration
    if not TRAIN_SAMPLE.exists():
        print("Creating sample dataset...")
        create_sample()
    
    # Load sample
    df = load_data_pandas(TRAIN_SAMPLE)
    print(f"\nLoaded sample: {df.shape}")
    print(f"\nFirst rows:\n{df.head()}")
    print(f"\nInfo:\n{df.info()}")
