"""
Model Training and Evaluation Module

Comprehensive model training with:
- Multiple ML algorithms (Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Cross-validation
- Hyperparameter tuning with Optuna
- Model evaluation and comparison
- Model persistence
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import logging
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.config import (
    RANDOM_STATE, CV_FOLDS, N_JOBS,
    MODEL_PARAMS, MODELS_TO_TRAIN
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModel:
    """
    Machine Learning model trainer and evaluator.
    """
    
    def __init__(self, model_type: str = 'lightgbm', random_state: int = RANDOM_STATE):
        """
        Initialize model.
        
        Args:
            model_type: Type of model ('ridge', 'random_forest', 'gradient_boosting', 
                       'xgboost', 'lightgbm')
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = []
        self.feature_importance = None
        self.training_history = {}
        self.cv_scores = {}
        
    def get_model(self, params: Optional[Dict] = None):
        """
        Get model instance with parameters.
        
        Args:
            params: Model hyperparameters
            
        Returns:
            Model instance
        """
        if params is None:
            params = MODEL_PARAMS.get(self.model_type, {})
        
        if self.model_type == 'ridge':
            return Ridge(**params, random_state=self.random_state)
        
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(**params, random_state=self.random_state)
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(**params, random_state=self.random_state)
        
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(**params, random_state=self.random_state)
        
        elif self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(**params, random_state=self.random_state)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             params: Optional[Dict] = None,
             eval_set: Optional[List[Tuple]] = None) -> Dict:
        """
        Train model on training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            params: Model hyperparameters
            eval_set: Validation set for early stopping [(X_val, y_val)]
            
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        self.feature_names = X_train.columns.tolist()
        self.model = self.get_model(params)
        
        # Train with early stopping if supported
        if self.model_type in ['xgboost', 'lightgbm'] and eval_set is not None:
            if self.model_type == 'xgboost':
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            else:  # lightgbm
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=[lgb.log_evaluation(period=0)]
                )
        else:
            self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        y_pred = self.model.predict(X_train)
        metrics = self._calculate_metrics(y_train, y_pred, 'train')
        
        # Store feature importance
        self._extract_feature_importance()
        
        logger.info(f"Training complete. RMSE: {metrics['rmse']:.2f}, "
                   f"MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, 
                set_name: str = 'test') -> Dict:
        """
        Evaluate model on data.
        
        Args:
            X: Features
            y: Target
            set_name: Name of dataset ('train', 'val', 'test')
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating on {set_name} set...")
        
        y_pred = self.predict(X)
        metrics = self._calculate_metrics(y, y_pred, set_name)
        
        logger.info(f"Evaluation complete. RMSE: {metrics['rmse']:.2f}, "
                   f"MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      cv: int = CV_FOLDS) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            CV metrics
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # RMSE (negative MSE)
        neg_mse_scores = cross_val_score(
            self.get_model(), X, y,
            cv=kf, scoring='neg_mean_squared_error',
            n_jobs=N_JOBS
        )
        rmse_scores = np.sqrt(-neg_mse_scores)
        
        # MAE
        neg_mae_scores = cross_val_score(
            self.get_model(), X, y,
            cv=kf, scoring='neg_mean_absolute_error',
            n_jobs=N_JOBS
        )
        mae_scores = -neg_mae_scores
        
        # R²
        r2_scores = cross_val_score(
            self.get_model(), X, y,
            cv=kf, scoring='r2',
            n_jobs=N_JOBS
        )
        
        self.cv_scores = {
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'cv_folds': cv
        }
        
        logger.info(f"CV Results - RMSE: {self.cv_scores['rmse_mean']:.2f} "
                   f"(±{self.cv_scores['rmse_std']:.2f}), "
                   f"R²: {self.cv_scores['r2_mean']:.4f} "
                   f"(±{self.cv_scores['r2_std']:.4f})")
        
        return self.cv_scores
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                          set_name: str) -> Dict:
        """Calculate regression metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'set': set_name
        }
        
        return metrics
    
    def _extract_feature_importance(self):
        """Extract feature importance from model."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            logger.warning("Model does not support feature importance")
            self.feature_importance = None
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available")
        
        return self.feature_importance.head(top_n)
    
    def save(self, model_path: str, metadata_path: str):
        """
        Save model and metadata to disk.
        
        Args:
            model_path: Path to save model
            metadata_path: Path to save metadata
        """
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'cv_scores': self.cv_scores,
            'training_date': datetime.now().isoformat(),
            'random_state': self.random_state,
            'model_params': self.model.get_params() if hasattr(self.model, 'get_params') else {}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load(self, model_path: str, metadata_path: str):
        """
        Load model and metadata from disk.
        
        Args:
            model_path: Path to model file
            metadata_path: Path to metadata file
        """
        self.model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.model_type = metadata['model_type']
        self.feature_names = metadata['feature_names']
        self.cv_scores = metadata['cv_scores']
        self.random_state = metadata['random_state']
        
        logger.info(f"Model loaded from {model_path}")


class HyperparameterTuner:
    """
    Hyperparameter optimization using Optuna.
    """
    
    def __init__(self, model_type: str, n_trials: int = 50, 
                 random_state: int = RANDOM_STATE):
        """
        Initialize tuner.
        
        Args:
            model_type: Type of model to tune
            n_trials: Number of optimization trials
            random_state: Random seed
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params = None
        self.study = None
    
    def objective(self, trial: optuna.Trial, X: pd.DataFrame, 
                 y: pd.Series) -> float:
        """
        Objective function for Optuna.
        
        Args:
            trial: Optuna trial
            X: Features
            y: Target
            
        Returns:
            RMSE score (to minimize)
        """
        # Define hyperparameter search space
        if self.model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'n_jobs': N_JOBS
            }
        
        elif self.model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            }
        
        elif self.model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)
            }
        
        else:
            raise ValueError(f"Tuning not supported for {self.model_type}")
        
        # Cross-validation
        model = MLModel(self.model_type, self.random_state)
        model.model = model.get_model(params)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        neg_mse_scores = cross_val_score(
            model.model, X, y,
            cv=kf, scoring='neg_mean_squared_error',
            n_jobs=1  # Optuna parallelizes trials
        )
        
        rmse = np.sqrt(-neg_mse_scores.mean())
        
        return rmse
    
    def tune(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Best parameters
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_type} "
                   f"({self.n_trials} trials)...")
        
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler
        )
        
        self.study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        
        logger.info(f"Tuning complete. Best RMSE: {self.study.best_value:.2f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params


def compare_models(X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series,
                  models: List[str] = MODELS_TO_TRAIN) -> pd.DataFrame:
    """
    Train and compare multiple models.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: List of model types to compare
        
    Returns:
        DataFrame with comparison results
    """
    logger.info(f"Comparing {len(models)} models...")
    
    results = []
    
    for model_type in models:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type}...")
        logger.info(f"{'='*50}")
        
        # Train model
        model = MLModel(model_type)
        train_metrics = model.train(X_train, y_train)
        
        # Evaluate on test set
        test_metrics = model.evaluate(X_test, y_test, 'test')
        
        # Cross-validation
        cv_metrics = model.cross_validate(X_train, y_train)
        
        # Compile results
        results.append({
            'model': model_type,
            'train_rmse': train_metrics['rmse'],
            'train_mae': train_metrics['mae'],
            'train_r2': train_metrics['r2'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2'],
            'cv_rmse_mean': cv_metrics['rmse_mean'],
            'cv_rmse_std': cv_metrics['rmse_std'],
            'cv_r2_mean': cv_metrics['r2_mean'],
            'cv_r2_std': cv_metrics['r2_std']
        })
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('cv_rmse_mean')
    
    logger.info("\n" + "="*70)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("="*70)
    logger.info(comparison_df.to_string(index=False))
    
    return comparison_df
