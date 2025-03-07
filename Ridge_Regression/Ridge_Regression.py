import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

df_rna = pd.read_hdf('train_cite_inputs.h5')
df_y = pd.read_hdf('train_cite_targets.h5')

variances = df_rna.var().sort_values(ascending=False)
top_features = variances.head(4096).index

X_filtered = df_rna[top_features]
X_train, X_test, y_train, y_test = train_test_split(X_filtered, df_y, test_size=0.2, random_state=42)
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.values
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.values

ridge_model = Ridge(alpha=1.0)
start_time = time.time()
scores = cross_val_score(ridge_model, X_train, y_train, cv=5, scoring='r2')
end_time = time.time()

print(f"Training completed in {end_time - start_time:.2f} seconds.")
print(f"Cross-Validation R^2 Scores: {scores}")
print(f"Average R^2 Score: {np.mean(scores)}")

ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
print(y_test.shape,y_pred.shape)
correlations = []
for i in range(y_test.shape[1]):
    correlation, _ = pearsonr(y_test[:, i], y_pred[:, i])
    correlations.append(correlation)

avg_correlation = np.mean(correlations)
print(f"Final Test Set Pearson Correlation Score: {avg_correlation}")