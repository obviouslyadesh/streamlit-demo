import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the dataset for feature extraction
df = pd.read_csv("pldata.csv")
df = df[['Team', 'Poss', 'SoT', 'Red Card', 'Recent Form', 'touch_in_opponent_box', 'Venue', 'GF']].dropna()
df['Venue'] = df['Venue'].astype('category').cat.codes

# Dictionary of team logos from Wikimedia
team_logos = {
    "Arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
    "AstonVilla": "https://upload.wikimedia.org/wikipedia/en/thumb/9/9a/Aston_Villa_FC_new_crest.svg/1200px-Aston_Villa_FC_new_crest.svg.png",
    "Bournemouth": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS2M_4dGiucAU0Oxi9Ech-tg_aSm4JkPXZLdQ&s",
    "Brentford": "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg",
    "BrightonandHoveAlbion": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
    "Burnley": "https://seeklogo.com/images/B/burnley-fc-logo-D08E749A01-seeklogo.com.png",
    "Chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
    "CrystalPalace": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTaLRov7Uq8QRyWD11FNid02Qxk_xMTxNKFbQ&s",
    "Everton": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",
    "Fulham": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",
    "Liverpool": "https://w7.pngwing.com/pngs/119/992/png-transparent-anfield-liverpool-f-c-liverpool-l-f-c-real-madrid-c-f-premier-league-premier-league-text-label-logo-thumbnail.png",
    "LutonTown": "https://upload.wikimedia.org/wikipedia/en/9/9d/Luton_Town_logo.svg",
    "ManchesterCity": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTi8y1XVOuSeygwE5T_-cWjjpiT7aZ_oPFhdA&s",
    "ManchesterUnited": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    "NewcastleUnited": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
    "NottinghamForest": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSHTxJRchnDE1K_KcDqvC08TFHIg55_kLYzKA&s",
    "SheffieldUnited": "https://upload.wikimedia.org/wikipedia/en/thumb/9/9c/Sheffield_United_FC_logo.svg/1200px-Sheffield_United_FC_logo.svg.png",
    "TottenhamHotspur": "https://e7.pngegg.com/pngimages/297/393/png-clipart-tottenham-hotspur-f-c-premier-league-northumberland-development-project-tottenham-hotspur-foundation-football-premier-league-logo-sports-thumbnail.png",
    "WestHamUnited": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
    "WolverhamptonWanderers": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg"
}


# Streamlit app title
st.title("Premier League Match Score Predictor")

# Select teams
teams = df['Team'].unique().tolist()
team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)

# Display team logos
col1, col2 = st.columns(2)
with col1:
    if team1 in team_logos:
        st.image(team_logos[team1], width=150)
    st.write(f"**{team1}**")

with col2:
    if team2 in team_logos:
        st.image(team_logos[team2], width=150)
    st.write(f"**{team2}**")

# Select venue
venue = st.radio("Who is playing at home?", (team1, team2))

# Function to predict the match result
def predict_match(team1, team2, venue, model, df):
    team1_data = df[df['Team'] == team1].mean(numeric_only=True)
    team2_data = df[df['Team'] == team2].mean(numeric_only=True)
    
    if venue == team1:
        team1_data['Venue'] = 1  # Home
        team2_data['Venue'] = 0  # Away
    else:
        team1_data['Venue'] = 0  # Away
        team2_data['Venue'] = 1  # Home
    
    features = ['Poss', 'SoT', 'Red Card', 'Recent Form', 'touch_in_opponent_box', 'Venue']
    team1_data = team1_data[features]
    team2_data = team2_data[features]

    team1_pred = model.predict(team1_data.values.reshape(1, -1))[0]
    team2_pred = model.predict(team2_data.values.reshape(1, -1))[0]
    
    return f"Predicted Score: {team1} {round(team1_pred)} - {round(team2_pred)} {team2}"

# Predict button
if st.button("Predict Match Outcome"):
    result = predict_match(team1, team2, venue, model, df)
    st.success(result)
