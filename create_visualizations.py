"""
Visualization script for model comparison and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = [15, 10]

print("Loading data and training models...")

# Load data
df = pd.read_csv('USA_Housing.csv')
df_clean = df.drop('Address', axis=1)

# Remove extreme outliers
Q1 = df_clean.quantile(0.25)
Q3 = df_clean.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ~((df_clean < (Q1 - 3 * IQR)) | (df_clean > (Q3 + 3 * IQR))).any(axis=1)
df_clean = df_clean[outlier_mask].reset_index(drop=True)

# Feature engineering
X = df_clean.drop('Price', axis=1)
y = df_clean['Price']

X['Income_per_Room'] = X['Avg. Area Income'] / X['Avg. Area Number of Rooms']
X['Bedrooms_to_Rooms_Ratio'] = X['Avg. Area Number of Bedrooms'] / X['Avg. Area Number of Rooms']
X['Income_per_Person'] = X['Avg. Area Income'] / X['Area Population']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# Train models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train_scaled)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test_scaled, y_pred)
    results[name] = {'model': model, 'y_pred': y_pred, 'r2': r2}

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. Model Comparison - R² Scores
ax1 = plt.subplot(2, 3, 1)
model_names = list(results.keys())
r2_scores = [results[m]['r2'] for m in model_names]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax1.bar(model_names, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison (R² Score)', fontsize=14, fontweight='bold')
ax1.set_ylim([0.85, 0.95])
ax1.grid(axis='y', alpha=0.3)
for i, (bar, score) in enumerate(zip(bars, r2_scores)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.4f}\n({score*100:.2f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. Actual vs Predicted - Linear Regression
ax2 = plt.subplot(2, 3, 2)
lr_pred = results['Linear Regression']['y_pred']
ax2.scatter(y_test_scaled, lr_pred, alpha=0.5, s=30, color='#3498db', edgecolor='black', linewidth=0.5)
ax2.plot([y_test_scaled.min(), y_test_scaled.max()], 
         [y_test_scaled.min(), y_test_scaled.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Price (Scaled)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Predicted Price (Scaled)', fontsize=12, fontweight='bold')
ax2.set_title('Linear Regression: Actual vs Predicted', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Residuals Distribution - Linear Regression
ax3 = plt.subplot(2, 3, 3)
residuals = y_test_scaled - lr_pred
ax3.hist(residuals, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Residuals', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
ax3.grid(alpha=0.3)

# 4. Feature Importance - Random Forest
ax4 = plt.subplot(2, 3, 4)
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

ax4.barh(feature_importance['feature'], feature_importance['importance'], 
         color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax4.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# 5. Cross-Validation Scores
ax5 = plt.subplot(2, 3, 5)
cv_results = {}
for name, result in results.items():
    cv_scores = cross_val_score(result['model'], X_train_scaled, y_train_scaled, cv=5, scoring='r2')
    cv_results[name] = cv_scores

positions = np.arange(len(cv_results))
bp = ax5.boxplot([cv_results[name] for name in model_names], 
                   labels=model_names, 
                   patch_artist=True,
                   widths=0.6)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax5.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax5.set_title('5-Fold Cross-Validation Scores', fontsize=14, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 6. Prediction Error Distribution
ax6 = plt.subplot(2, 3, 6)
# Convert back to original scale for meaningful interpretation
y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
lr_pred_original = scaler_y.inverse_transform(lr_pred.reshape(-1, 1)).ravel()
errors = y_test_original - lr_pred_original
error_pct = (errors / y_test_original) * 100

ax6.hist(error_pct, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
ax6.axvline(x=0, color='green', linestyle='--', linewidth=2)
ax6.set_xlabel('Prediction Error (%)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax6.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
ax6.grid(alpha=0.3)

mean_error = np.mean(np.abs(error_pct))
ax6.text(0.05, 0.95, f'Mean Absolute Error: {mean_error:.2f}%', 
         transform=ax6.transAxes, fontsize=11, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('model_analysis_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'model_analysis_visualization.png'")
plt.close()

# Create additional detailed comparison plot
fig2, axes = plt.subplots(1, 3, figsize=(20, 5))

for idx, (name, color) in enumerate(zip(model_names, colors)):
    y_pred = results[name]['y_pred']
    r2 = results[name]['r2']
    
    axes[idx].scatter(y_test_scaled, y_pred, alpha=0.5, s=30, color=color, edgecolor='black', linewidth=0.5)
    axes[idx].plot([y_test_scaled.min(), y_test_scaled.max()], 
                   [y_test_scaled.min(), y_test_scaled.max()], 
                   'r--', lw=2, label='Perfect Prediction')
    axes[idx].set_xlabel('Actual Price (Scaled)', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Predicted Price (Scaled)', fontsize=12, fontweight='bold')
    axes[idx].set_title(f'{name}\nR² = {r2:.4f} ({r2*100:.2f}%)', fontsize=13, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('detailed_model_comparison.png', dpi=300, bbox_inches='tight')
print("Detailed comparison saved as 'detailed_model_comparison.png'")
plt.close()

print("\nAll visualizations created successfully!")
