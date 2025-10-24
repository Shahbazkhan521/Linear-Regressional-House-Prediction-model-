"""
Improved House Price Prediction Model
This script addresses all weaknesses and implements best practices for ML
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*60)
print("IMPROVED HOUSE PRICE PREDICTION MODEL")
print("="*60)

# 1. Load and Explore Data
print("\n1. LOADING DATA...")
df = pd.read_csv('USA_Housing.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

# 2. Data Preprocessing
print("\n2. DATA PREPROCESSING...")
# Drop Address column as it's not useful for prediction
df_clean = df.drop('Address', axis=1)

# Check for outliers using IQR method
print("\n2.1 Outlier Detection (IQR Method):")
Q1 = df_clean.quantile(0.25)
Q3 = df_clean.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df_clean < (Q1 - 1.5 * IQR)) | (df_clean > (Q3 + 1.5 * IQR))).sum()
print(f"Outliers per column:\n{outliers}")

# Remove extreme outliers (optional - comment out if you want to keep all data)
# For now, we'll keep the data but note the outliers
outlier_mask = ~((df_clean < (Q1 - 3 * IQR)) | (df_clean > (Q3 + 3 * IQR))).any(axis=1)
df_clean = df_clean[outlier_mask].reset_index(drop=True)
print(f"Dataset shape after removing extreme outliers: {df_clean.shape}")

# 3. Feature Engineering
print("\n3. FEATURE ENGINEERING...")
X = df_clean.drop('Price', axis=1)
y = df_clean['Price']

# Create additional features
X['Income_per_Room'] = X['Avg. Area Income'] / X['Avg. Area Number of Rooms']
X['Bedrooms_to_Rooms_Ratio'] = X['Avg. Area Number of Bedrooms'] / X['Avg. Area Number of Rooms']
X['Income_per_Person'] = X['Avg. Area Income'] / X['Area Population']
print(f"Features after engineering: {X.columns.tolist()}")

# 4. Train-Test Split
print("\n4. SPLITTING DATA...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 5. Feature Scaling (CORRECT WAY)
print("\n5. FEATURE SCALING...")
# Scale features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)  # Use same scaler, don't fit again!

# Scale target (using the SAME scaler for train and test)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()  # CORRECT!

print("Scaling completed successfully!")
print(f"X_train mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
print(f"y_train mean: {y_train_scaled.mean():.6f}, std: {y_train_scaled.std():.6f}")

# 6. Model Training and Evaluation
print("\n6. MODEL TRAINING AND EVALUATION...")
print("="*60)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions (CORRECT - using each model's own predictions!)
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train_scaled, y_pred_train)
    test_r2 = r2_score(y_test_scaled, y_pred_test)
    test_mse = mean_squared_error(y_test_scaled, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_scaled, y_pred_test)
    
    # Cross-validation score (5-fold)
    cv_scores = cross_val_score(model, X_train_scaled, y_train_scaled, 
                                cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'y_pred': y_pred_test
    }
    
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    print(f"  CV R² (5-fold): {cv_mean:.4f} (+/- {cv_std:.4f})")

# 7. Hyperparameter Tuning for Best Model
print("\n7. HYPERPARAMETER TUNING...")
print("="*60)

# Tune Random Forest (usually performs well)
print("\nTuning Random Forest Regressor...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train_scaled)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV R² score: {grid_search.best_score_:.4f}")

# Evaluate tuned model
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_scaled)
best_r2 = r2_score(y_test_scaled, y_pred_best)
best_rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_best))
best_mae = mean_absolute_error(y_test_scaled, y_pred_best)

print(f"\nTuned Random Forest Performance:")
print(f"  Test R²: {best_r2:.4f}")
print(f"  Test RMSE: {best_rmse:.4f}")
print(f"  Test MAE: {best_mae:.4f}")

results['Tuned Random Forest'] = {
    'model': best_rf,
    'test_r2': best_r2,
    'test_rmse': best_rmse,
    'test_mae': best_mae,
    'y_pred': y_pred_best
}

# 8. Feature Importance Analysis
print("\n8. FEATURE IMPORTANCE ANALYSIS...")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# 9. Results Summary
print("\n9. FINAL RESULTS SUMMARY")
print("="*60)

summary_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test R²': [results[m]['test_r2'] for m in results.keys()],
    'Test RMSE': [results[m]['test_rmse'] for m in results.keys()],
    'Test MAE': [results[m]['test_mae'] for m in results.keys()]
}).sort_values('Test R²', ascending=False)

print("\n" + summary_df.to_string(index=False))

best_model_name = summary_df.iloc[0]['Model']
best_model_r2 = summary_df.iloc[0]['Test R²']

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_model_name}")
print(f"R² Score: {best_model_r2:.4f} ({best_model_r2*100:.2f}% accuracy)")
print(f"{'='*60}")

# 10. Save predictions for visualization
print("\n10. SAVING RESULTS...")

# Convert predictions back to original scale
y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
y_pred_original = scaler_y.inverse_transform(y_pred_best.reshape(-1, 1)).ravel()

predictions_df = pd.DataFrame({
    'Actual Price': y_test_original,
    'Predicted Price': y_pred_original,
    'Difference': y_test_original - y_pred_original,
    'Percentage Error': abs((y_test_original - y_pred_original) / y_test_original * 100)
})

predictions_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)

# Summary of improvements made
print("\nIMPROVEMENTS IMPLEMENTED:")
print("✓ Added dataset file")
print("✓ Fixed target scaling (same scaler for train and test)")
print("✓ Fixed model evaluation (each model uses its own predictions)")
print("✓ Added cross-validation (5-fold)")
print("✓ Implemented hyperparameter tuning (GridSearchCV)")
print("✓ Added feature engineering (3 new features)")
print("✓ Implemented outlier detection and removal")
print("✓ Added feature importance analysis")
print("✓ Added comprehensive evaluation metrics")
print("✓ Added multiple regression models for comparison")
print("✓ Improved code structure and documentation")
