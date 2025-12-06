import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def train_ridge(self, X_train, y_train):
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
        
        model = Ridge(random_state=42)
        grid_search = GridSearchCV(
            model, param_grid, cv=5, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.models['ridge'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def train_random_forest(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            model, param_grid, cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.models['random_forest'] = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def train_catboost(self, X_train, y_train, cat_features_indices=None):
        param_grid = {
            'iterations': [500, 1000],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.03, 0.05],
            'l2_leaf_reg': [1, 3, 5]
        }
        
        model = CatBoostRegressor(
            random_state=42,
            verbose=False,
            cat_features=cat_features_indices
        )
        
        grid_search = GridSearchCV(
            model, param_grid, cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.models['catboost'] = grid_search.best_estimator_
        self.best_model = grid_search.best_estimator_
        return grid_search.best_estimator_
    
    def evaluate_models(self, X_test, y_test):
        results = []
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'model': name,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            })
            
            self.results[name] = {
                'predictions': y_pred,
                'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2}
            }
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('mae')
        
        self.best_model_name = results_df.iloc[0]['model']
        self.best_model = self.models[self.best_model_name]
        
        return results_df
    
    def get_feature_importance(self, model_name, feature_names):
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не обучена")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df
    
    def save_model(self, model_name, path):
        if model_name in self.models:
            joblib.dump(self.models[model_name], path)
            print(f"Модель {model_name} сохранена в {path}")
    
    def save_best_model(self, path):
        if self.best_model is not None:
            joblib.dump(self.best_model, path)
            print(f"Лучшая модель сохранена в {path}")
