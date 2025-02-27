import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    best_model = pickle.load(file)

# Load dataset
df = pd.read_csv("pldata.csv")

# Preprocessing
df = df[['Team', 'Poss', 'SoT', 'Red Card', 'Recent Form', 'touch_in_opponent_box', 'Venue', 'GF']].dropna()
df['Venue'] = df['Venue'].astype('category').cat.codes

# Define feature matrix (X) and target variable (y)
X = df[['Poss', 'SoT', 'Red Card', 'Recent Form', 'touch_in_opponent_box', 'Venue']]
y = df['GF']

# Perform 5-Fold Cross-Validation
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = -cross_val_score(best_model, X, y, scoring='neg_mean_squared_error', cv=kf)
mae_scores = -cross_val_score(best_model, X, y, scoring='neg_mean_absolute_error', cv=kf)
r2_scores = cross_val_score(best_model, X, y, scoring='r2', cv=kf)
rmse_scores = np.sqrt(mse_scores)

# Create results DataFrame
results_df = pd.DataFrame({
    'Fold': range(1, 6),
    'MSE': mse_scores,
    'MAE': mae_scores,
    'R²': r2_scores,
    'RMSE': rmse_scores
})

# Streamlit UI
st.title("Model Performance Evaluation")
st.write("### 5-Fold Cross-Validation Results")
st.dataframe(results_df.style.format({"MSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}", "RMSE": "{:.4f}"}))
