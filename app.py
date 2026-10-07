import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium

# ---------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------------------------------------
st.set_page_config(
page_title="Génie Civil - Fiabilité & Algorithmes Probabilistes",
page_icon="🎲",
layout="wide"
)

st.title("🎲 Surveillance & Analyse Probabiliste de Fiabilité des Ouvrages")

# Initialisation des données de simulation en session
if "donnees_probas" not in st.session_state:
st.session_state.donnees_probas = pd.DataFrame([
{"Ouvrage": "Pont A1", "Type": "Pont", "Tassement_moyen": 8.0, "Ecart_type": 2.0, "Seuil_Rupture": 12.0, "Latitude": 48.8566, "Longitude": 2.3522},
{"Ouvrage": "Viaduc B2", "Type": "Viaduc", "Tassement_moyen": 11.5, "Ecart_type": 3.5, "Seuil_Rupture": 15.0, "Latitude": 48.8600, "Longitude": 2.3400},
{"Ouvrage": "Mur C3", "Type": "Soutènement", "Tassement_moyen": 4.0, "Ecart_type": 1.0, "Seuil_Rupture": 10.0, "Latitude": 48.8500, "Longitude": 2.3600}
    ])

# ---------------------------------------------------------
# BARRE LATÉRALE : SAISIE DES PARAMÈTRES STOCHASTIQUES
# ---------------------------------------------------------
st.sidebar.header("⚙️ Paramètres Stochastiques")

with st.sidebar.form("form_probabiliste"):
    nom_ouvrage = st.text_input("Nom de l'ouvrage", value="Pont Sur l'Ogooué")
    type_ouvrage = st.selectbox("Type", ["Pont", "Viaduc", "Mur de Soutènement", "Fondation"])
    
    col_gps1, col_gps2 = st.columns(2)
    lat = col_gps1.number_input("Lat", value=48.8550, format="%.4f")
    lon = col_gps2.number_input("Lon", value=2.3500, format="%.4f")

    st.subheader("Distribution du Tassement (Loi Normale)")
    mu_tassement = st.number_input("Tassement Moyen (μ en mm)", value=7.5, step=0.5)
    sigma_tassement = st.number_input("Écart-type (σ en mm)", value=2.5, step=0.1)
    
    st.subheader("Critère de Défaillance")
    seuil_critique = st.number_input("Tassement Limite Admise (S_lim en mm)", value=12.0, step=0.5)

    n_simulations = st.sidebar.select_slider(
        "Nombre de tirages Monte-Carlo (N)",
        options=[1000, 5000, 10000, 50000, 100000],
        value=10000
    )

    submit = st.form_submit_button("Ajouter & Lancer la simulation")

if submit:
    nouvelle_entree = pd.DataFrame([{
        "Ouvrage": nom_ouvrage, "Type": type_ouvrage,
        "Tassement_moyen": mu_tassement, "Ecart_type": sigma_tassement,
        "Seuil_Rupture": seuil_critique, "Latitude": lat, "Longitude": lon
    }])
    st.session_state.donnees_probas = pd.concat([st.session_state.donnees_probas, nouvelle_entree], ignore_index=True)

# ---------------------------------------------------------
# SÉLECTION ET MOTEUR DE SIMULATION MONTE-CARLO
# ---------------------------------------------------------
df = st.session_state.donnees_probas
ouvrage_cible = st.selectbox("🎯 Sélectionner un ouvrage pour l'Analyse de Fiabilité :", df["Ouvrage"].unique())

row = df[df["Ouvrage"] == ouvrage_cible].iloc[0]

# --- Algorithme probabiliste : Simulation de Monte-Carlo ---
# Génération de N variables aléatoires caractérisant le comportement de l'ouvrage
np.random.seed(42)  # Reproductibilité
tassements_simules = np.random.normal(loc=row["Tassement_moyen"], scale=row["Ecart_type"], size=n_simulations)

# Calcul de la Fonction de Marge de Sécurité Z = S_lim - S
marge_securite = row["Seuil_Rupture"] - tassements_simules

# Identification des cas de défaillance (Z < 0)
defaillances = tassements_simules > row["Seuil_Rupture"]
probabilite_defaillance = (np.sum(defaillances) / n_simulations) * 100

# Indice de fiabilité β (Hasofer-Lind / Cornell)
beta = (row["Seuil_Rupture"] - row["Tassement_moyen"]) / row["Ecart_type"]

# ---------------------------------------------------------
# TABLEAU DE BORD FIABILISTE
# ---------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Nombre de Tirages (Monte-Carlo)", f"{n_simulations:,}")
m2.metric("Indice de Fiabilité (β)", f"{beta:.2f}")
m3.metric("Probabilité de Défaillance (Pf)", f"{probabilite_defaillance:.2f} %", 
          delta="Élevée" if probabilite_defaillance > 5 else "Acceptable", delta_color="inverse")
m4.metric("Seuil Limite (S_lim)", f"{row['Seuil_Rupture']} mm")

st.divider()

tab_proba, tab_carte, tab_comparatif = st.tabs(["📉 Simulation Probabiliste", "🗺️ Cartographie des Risques", "📊 Vue d'ensemble des Ouvrages"])

# 1. Graphique de la distribution probabiliste
with tab_proba:
    st.subheader(f"Analyse par Simulation de Monte-Carlo — {ouvrage_cible}")
    
    # Histogramme de la distribution des tassements
    fig = go.Figure()
    
    # Zone des tirages conformes
    fig.add_trace(go.Histogram(
        x=tassements_simules[~defaillances],
        name="Comportement Conforme (Z > 0)",
        marker_color="skyblue",
        opacity=0.75,
        nbinsx=80
    ))
    
    # Zone de défaillance (Rupture/Tassement excessif)
    fig.add_trace(go.Histogram(
        x=tassements_simules[defaillances],
        name="Zone de Défaillance (Z ≤ 0)",
        marker_color="crimson",
        opacity=0.85,
        nbinsx=80
    ))
    
    # Ligne verticale indiquant le seuil
    fig.add_vline(
        x=row["Seuil_Rupture"], line_width=3, line_dash="dash", line_color="red",
        annotation_text=f"Seuil Critique ({row['Seuil_Rupture']} mm)", annotation_position="top right"
    )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Tassement Simulé (mm)",
        yaxis_title="Fréquence / Nombre d'événements",
        title=f"Distribution de la réponse structurale ($N={n_simulations}$ tirages)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 2. Carte interactive basée sur l'indice de risque probabiliste
with tab_carte:
    st.subheader("Cartographie pondérée par la Probabilité de Défaillance")
    
    carte = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=12)
    
    for _, item in df.iterrows():
        # Calcul rapide de Pf pour la carte
        pf_item = (1 - np.exp(-((item["Seuil_Rupture"] - item["Tassement_moyen"])/item["Ecart_type"]))) * 100
        couleur = "red" if pf_item > 10 else ("orange" if pf_item > 2 else "green")
        
        folium.CircleMarker(
            location=[item["Latitude"], item["Longitude"]],
            radius=8 + (pf_item / 2),  # La taille dépend du risque
            color=couleur,
            fill=True,
            fill_color=couleur,
            popup=f"<b>{item['Ouvrage']}</b><br>Moyenne: {item['Tassement_moyen']} mm<br>σ: {item['Ecart_type']}<br>β: {((item['Seuil_Rupture'] - item['Tassement_moyen'])/item['Ecart_type']):.2f}"
        ).add_to(carte)
        
    st_folium(carte, width=1000, height=450)

# 3. Synthèse globale
with tab_comparatif:
    st.subheader("Données d'Entrée & Indicateurs de Fiabilité")
    st.dataframe(df, use_container_width=True)
