import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.preprocessor = None
        
    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        print(f"Данные загружены. Размер: {self.data.shape}")
        return self.data
    
    def clean_data(self, min_minutes=900):
        if self.data is None:
            self.load_data()
            
        df = self.data.copy()
        
        df = df[df['minutes'] >= min_minutes]
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
        
        df = df.dropna(subset=['market_value'])
        
        df = df[df['market_value'] > 0]
        
        if 'age' in df.columns:
            df = df[df['age'] >= 16]
            df = df[df['age'] <= 40]
        
        Q1 = df['market_value'].quantile(0.01)
        Q3 = df['market_value'].quantile(0.99)
        df = df[(df['market_value'] >= Q1) & (df['market_value'] <= Q3)]
        
        self.data = df.reset_index(drop=True)
        print(f"Данные очищены. Новый размер: {self.data.shape}")
        return self.data
    
    def prepare_features(self, target_col='market_value'):
        if self.data is None:
            raise ValueError("Данные не загружены. Сначала вызовите load_data() или clean_data()")
        
        df = self.data.copy()
        
        y = np.log1p(df[target_col])
        
        features_to_drop = ['player', 'club', 'league', 'season', target_col]
        X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
        
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        X_processed = self.preprocessor.fit_transform(X)
        
        feature_names = numeric_cols.copy()
        if categorical_cols:
            cat_features = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
            feature_names.extend(cat_features)
        
        return X_processed, y.values, feature_names, df
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        X, y, feature_names, df = self.prepare_features()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        train_idx, test_idx = train_test_split(
            df.index, test_size=test_size, random_state=random_state
        )
        
        train_data = df.loc[train_idx].copy()
        test_data = df.loc[test_idx].copy()
        
        return X_train, X_test, y_train, y_test, feature_names, train_data, test_data
