import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import sys
import os

sys.path.append('src')

from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer

def main():
    print("=" * 60)
    print("Moneyball Football Analytics System")
    print("=" * 60)
    
    data_path = "data/raw/football_data.csv"
    
    if not os.path.exists(data_path):
        print(f"Ошибка: Файл данных не найден: {data_path}")
        print("Пожалуйста, поместите данные в data/raw/football_data.csv")
        return
    
    print("\n1. Загрузка и обработка данных...")
    preprocessor = DataPreprocessor(data_path)
    data = preprocessor.load_data()
    data = preprocessor.clean_data(min_minutes=900)
    
    print(f"\nРазмер данных после очистки: {data.shape}")
    print(f"Количество игроков: {len(data)}")
    print(f"Диапазон стоимости: {data['market_value'].min():.2f} - {data['market_value'].max():.2f} млн €")
    
    print("\n2. Подготовка признаков...")
    X_train, X_test, y_train, y_test, feature_names, train_data, test_data = preprocessor.get_train_test_split()
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Количество признаков: {len(feature_names)}")
    
    print("\n3. Обучение моделей...")
    trainer = ModelTrainer()
    
    print("  - Обучение Ridge регрессии...")
    ridge_model = trainer.train_ridge(X_train, y_train)
    
    print("  - Обучение Random Forest...")
    rf_model = trainer.train_random_forest(X_train, y_train)
    
    print("  - Обучение CatBoost...")
    catboost_model = trainer.train_catboost(X_train, y_train)
    
    print("\n4. Оценка моделей...")
    results_df = trainer.evaluate_models(X_test, y_test)
    
    print("\nРезультаты моделей:")
    print("-" * 50)
    for _, row in results_df.iterrows():
        print(f"{row['model']:15} MAE: {row['mae']:.4f}  R²: {row['r2']:.4f}")
    print("-" * 50)
    
    best_model_name = results_df.iloc[0]['model']
    print(f"\nЛучшая модель: {best_model_name}")
    
    print("\n5. Анализ важности признаков...")
    if best_model_name in ['catboost', 'random_forest']:
        importance_df = trainer.get_feature_importance(best_model_name, feature_names)
        if importance_df is not None:
            print("\nТоп-10 важных признаков:")
            print("-" * 40)
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"{i+1:2}. {row['feature'][:50]:50} {row['importance']:.4f}")
    
    print("\n6. Восстановление предсказаний в исходный масштаб...")
    best_predictions = trainer.results[best_model_name]['predictions']
    
    test_data['predicted_log'] = best_predictions
    test_data['predicted_value'] = np.expm1(best_predictions)
    test_data['actual_value'] = np.expm1(y_test)
    test_data['absolute_error'] = np.abs(test_data['predicted_value'] - test_data['actual_value'])
    
    mae_original = mean_absolute_error(
        test_data['actual_value'], 
        test_data['predicted_value']
    )
    
    print(f"\nСредняя абсолютная ошибка в млн €: {mae_original:.2f}")
    
    print("\n7. Примеры предсказаний:")
    print("-" * 60)
    sample_results = test_data[['player', 'club', 'actual_value', 'predicted_value', 'absolute_error']].head()
    for _, row in sample_results.iterrows():
        print(f"{row['player'][:20]:20} | {row['club'][:15]:15} | Факт: {row['actual_value']:6.1f} | Прогноз: {row['predicted_value']:6.1f} | Ошибка: {row['absolute_error']:4.1f}")
    
    print("\n8. Сохранение результатов...")
    test_data.to_csv("data/processed/predictions.csv", index=False)
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    trainer.save_best_model(f"models/{best_model_name}_best_model.pkl")
    
    print("\n" + "=" * 60)
    print("Анализ завершен успешно!")
    print("=" * 60)

if __name__ == "__main__":
    main()
