import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import os

MODEL_CHECKPOINT = "random_forest_checkpoint.pkl"

df_rna = pd.read_hdf('train_cite_inputs.h5', start= 0, stop= 10000)
df_y = pd.read_hdf('train_cite_targets.h5', start= 0, stop= 10000)

# Select high variance features
variances = df_rna.var().sort_values(ascending=False)
top_features = variances.head(4096).index
X_filtered = df_rna[top_features]

X_train, X_test, y_train, y_test = train_test_split(X_filtered, df_y, test_size=0.2, random_state=42)

y_train = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
y_test = y_test.values if isinstance(y_test, pd.DataFrame) else y_test

if os.path.exists(MODEL_CHECKPOINT):
    print("Loading existing model checkpoint...")
    random_forest_model = joblib.load(MODEL_CHECKPOINT)
else:
    print("No checkpoint found. Initializing a new model...")
    random_forest_model = RandomForestRegressor(n_estimators=5, warm_start=True, random_state=42)  # Start small

total_trees = 100 
trees_per_step = 5 

while random_forest_model.n_estimators < total_trees:
    print(f"Training Random Forest with {random_forest_model.n_estimators} trees...")

    # Train model
    start_time = time.time()
    random_forest_model.fit(X_train, y_train)
    end_time = time.time()

    # Save updated model checkpoint
    joblib.dump(random_forest_model, MODEL_CHECKPOINT)
    print(f"Model checkpoint saved at {MODEL_CHECKPOINT} after {random_forest_model.n_estimators} trees.")
    random_forest_model.n_estimators += trees_per_step

# Final model evaluation
print("Final model trained. Evaluating performance...")
y_pred = random_forest_model.predict(X_test)
correlations = [
    pearsonr(y_test[:, i], y_pred[:, i])[0] if np.std(y_test[:, i]) != 0 and np.std(y_pred[:, i]) != 0 else -1.0
    for i in range(y_test.shape[1])
]
avg_correlation = np.mean(correlations)
print(f"Final Test Set Pearson Correlation Score: {avg_correlation}")