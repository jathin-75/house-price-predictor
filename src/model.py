import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ── 1. Load data ──────────────────────────────────────────
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target
print("Dataset loaded:", df.shape)
print(df.head())

# ── 2. EDA ────────────────────────────────────────────────
print("\nMissing values:\n", df.isnull().sum())

plt.figure(figsize=(8, 4))
plt.hist(df['Price'], bins=50, color='steelblue', edgecolor='white')
plt.title('Distribution of House Prices')
plt.xlabel('Price (in $100,000s)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('data/price_distribution.png')
plt.show()

# ── 3. Preprocess ─────────────────────────────────────────
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Train & compare models ─────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree':     DecisionTreeRegressor(random_state=42),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
}

print("\nModel Results:")
print(f"{'Model':<25} | {'RMSE':>6} | {'R2':>6}")
print("-" * 42)
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"{name:<25} | {rmse:>6.4f} | {r2:>6.4f}")

# ── 5. Evaluate best model ────────────────────────────────
best_model = models['Random Forest']
y_pred = best_model.predict(X_test_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, s=10, color='purple')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.savefig('data/actual_vs_predicted.png')
plt.show()

# Feature importance
importances = pd.Series(
    best_model.feature_importances_,
    index=housing.feature_names
).sort_values(ascending=False)

importances.plot(kind='bar', color='steelblue', figsize=(8, 4))
plt.title('Feature Importance — Random Forest')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('data/feature_importance.png')
plt.show()

# ── 6. Save model ─────────────────────────────────────────
joblib.dump(best_model, 'src/house_price_model.pkl')
joblib.dump(scaler,     'src/scaler.pkl')
print("\nModel saved successfully!")