import pandas as pd
from sklearn.preprocessing import LabelEncoder
import shap
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Load the dataset (same dataset used during model training)
df = pd.read_csv('pldata.csv')

# Preprocessing: Select relevant columns and drop missing values
df = df[['Team', 'Poss', 'SoT', 'Red Card', 'Recent Form', 'touch_in_opponent_box', 'Venue', 'GF']].dropna()

# Convert categorical columns to numeric using Label Encoding (for example, 'Venue')
label_encoder = LabelEncoder()
df['Venue'] = label_encoder.fit_transform(df['Venue'])

# Define feature matrix (X)
X = df[['Poss', 'SoT', 'Red Card', 'Recent Form', 'touch_in_opponent_box', 'Venue']]

# ----- XGBoost Built-in Feature Importance -----
feature_importance = best_model.feature_importances_
feature_names = X.columns

# Display XGBoost Feature Importance
st.subheader('XGBoost Feature Importance')
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=feature_importance, y=feature_names, ax=ax)
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Feature')
ax.set_title('XGBoost Feature Importance')
st.pyplot(fig)

# ----- SHAP Feature Importance -----
st.subheader('SHAP Feature Importance')
explainer = shap.Explainer(best_model, X)
shap_values = explainer(X)

# Plot SHAP summary plot
fig, ax = plt.subplots(figsize=(10, 5))
shap.summary_plot(shap_values, X, show=False)  # show=False to prevent auto display
st.pyplot(fig)