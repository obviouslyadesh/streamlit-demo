import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import ttest_ind, pearsonr, f_oneway
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats

# Load dataset
file_path = "pldata.csv"
df = pd.read_csv(file_path)

# Convert 'Result' to binary (Win = 1, Loss/Draw = 0)
df['Win'] = df['Result'].apply(lambda x: 1 if x == 'W' else 0)

# Hypothesis 1: Playing at home increases chances of winning
st.subheader("Hypothesis 1: Playing at Home Increases Chances of Winning")

home_wins = df[df['Venue'] == 'Home']['Win']
away_wins = df[df['Venue'] == 'Away']['Win']
t_stat, p_value_home_away = ttest_ind(home_wins, away_wins)
st.write(f'T-test for Home vs Away Wins: T-statistic={t_stat}, P-value={p_value_home_away}')

# Visualize with countplot
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.countplot(x='Venue', hue='Result', data=df, ax=ax1)
ax1.set_title("Home vs Away Match Outcomes")
st.pyplot(fig1)

# Hypothesis 2: Higher possession leads to more shots on target
st.subheader("Hypothesis 2: Higher Possession Leads to More Shots on Target")

corr_poss_sot, p_value_poss_sot = pearsonr(df['Poss'], df['SoT'])
st.write(f'Correlation between Possession and Shots on Target: {corr_poss_sot}, P-value: {p_value_poss_sot}')

# Visualize with scatterplot
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=df['Poss'], y=df['SoT'], ax=ax2)
ax2.set_title('Possession vs Shots on Target')
ax2.set_xlabel('Possession (%)')
ax2.set_ylabel('Shots on Target')
st.pyplot(fig2)

# Hypothesis 3: Recent form is a strong predictor of match outcome
st.subheader("Hypothesis 3: Recent Form is a Strong Predictor of Match Outcome")

X = df[['Poss', 'SoT', 'Red Card']]
y = df['Win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.write(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_pred)}')

# Prepare data for visualization (box plot for recent form distribution)
df = df[['Recent Form', 'Result']].dropna()
win_form = df[df['Result'] == 'W']['Recent Form']
draw_form = df[df['Result'] == 'D']['Recent Form']
loss_form = df[df['Result'] == 'L']['Recent Form']

# Perform ANOVA test
f_stat, p_value = f_oneway(win_form, draw_form, loss_form)
st.write(f"ANOVA Results: F-statistic = {f_stat:.4f}, P-value = {p_value:.4e}")

# Box plot visualization
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.boxplot(x=df['Result'], y=df['Recent Form'], ax=ax3)
ax3.set_title("Recent Form Distribution Across Match Outcomes")
ax3.set_xlabel("Match Result")
ax3.set_ylabel("Recent Form")
st.pyplot(fig3)

# Hypothesis 4: Saves vs Goals Conceded
st.subheader("Hypothesis 4: Saves vs Goals Conceded")

fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.regplot(x=df['Saves'], y=df['GA'], scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax4)
ax4.set_title("Effect of Saves on Goals Conceded")
ax4.set_xlabel("Total Saves")
ax4.set_ylabel("Goals Conceded (GA)")
correlation1 = df[['Saves', 'GA']].corr().iloc[0,1]
plt.figtext(0.15, 0.8, f"Correlation: {correlation1:.2f}", fontsize=12, color='red')
st.pyplot(fig4)

# Hypothesis 5: Saves vs Total Shots Faced
st.subheader("Hypothesis 5: Saves vs Total Shots Faced")

fig5, ax5 = plt.subplots(figsize=(8, 5))
sns.regplot(x=df['Saves'], y=df['Sh'], scatter_kws={'alpha':0.6}, line_kws={'color':'blue'}, ax=ax5)
ax5.set_title("Effect of Saves on Shots Faced")
ax5.set_xlabel("Total Saves")
ax5.set_ylabel("Total Shots Faced (Sh)")
correlation2 = df[['Saves', 'Sh']].corr().iloc[0,1]
plt.figtext(0.15, 0.8, f"Correlation: {correlation2:.2f}", fontsize=12, color='blue')
st.pyplot(fig5)

# Hypothesis 6: Correlation between 'touch_in_opponent_box' and 'GF' (Goals Scored)
st.subheader("Hypothesis 6: Correlation between 'touch_in_opponent_box' and 'GF' (Goals Scored)")

x = df['touch_in_opponent_box']
y = df['GF']

# Calculate Pearson correlation coefficient and p-value
corr, p_value = pearsonr(x, y)

# Print the correlation coefficient and p-value
st.write(f"Pearson correlation coefficient between 'touch_in_opponent_box' and 'GF': {corr:.3f}")
st.write(f"P-value: {p_value:.3f}")

# Visualizing the relationship with a scatter plot
fig6, ax6 = plt.subplots(figsize=(8, 6))
ax6.scatter(x, y, alpha=0.5)
ax6.set_title("Relationship between 'touch_in_opponent_box' and 'GF' (Goals Scored)")
ax6.set_xlabel("Touch in Opponent's Box")
ax6.set_ylabel("Goals Scored (GF)")

# Interpretation
if p_value < 0.05:
    st.write("There is a significant correlation between 'touch_in_opponent_box' and 'GF'.")
else:
    st.write("There is no significant correlation between 'touch_in_opponent_box' and 'GF'.")

st.pyplot(fig6)
