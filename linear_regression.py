# linear_regression.py

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# Import + minimal preprocess
data = fetch_california_housing(as_frame=True)
df = data.frame  # features + target in 'MedHouseVal'
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Optional scaling for linear regression stability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Fit Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate
y_pred = lr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.4f}")

# Save metrics
with open("outputs/metrics.txt", "w") as f:
    f.write(f"MAE: {mae:.6f}\nMSE: {mse:.6f}\nR2: {r2:.6f}\n")
    f.write("Coefficients (feature -> weight):\n")
    for name, w in zip(X.columns, lr.coef_):
        f.write(f"{name}: {w:.6f}\n")
    f.write(f"Intercept: {lr.intercept_:.6f}\n")

# Plot regression line (simple 2D demo with one feature)
# Choose the most important feature by absolute coefficient
top_idx = int(np.argmax(np.abs(lr.coef_)))
feat = X.columns[top_idx]

# Build a scatter vs prediction line for the chosen feature
x_feat_train = X_train[:, top_idx]
x_feat_test = X_test[:, top_idx]

plt.figure(figsize=(7,5))
plt.scatter(x_feat_test, y_test, s=10, alpha=0.5, label="Actual")
plt.scatter(x_feat_test, y_pred, s=10, alpha=0.5, label="Predicted")
plt.title(f"Linear Regression: {feat} vs MedHouseVal (test set)")
plt.xlabel(f"{feat} (scaled)")
plt.ylabel("MedHouseVal")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/regression_scatter_pred_vs_actual.png", dpi=150)
plt.close()
