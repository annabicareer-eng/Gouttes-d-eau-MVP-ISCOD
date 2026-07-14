import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


DATABASE_PATH = "goutte_eau.db"
MODEL_PATH = "modele_pluie.joblib"
FEATURES_PATH = "features_modele.joblib"


@st.cache_resource
def charger_modele():
    modele = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return modele, features


@st.cache_data
def charger_stations():
    with sqlite3.connect(DATABASE_PATH) as connexion:
        return pd.read_sql_query(
            """
            SELECT DISTINCT
                geo_id_wmo,
                name
            FROM observations_meteo
            ORDER BY name
            """,
            connexion
        )


def determiner_saison(mois):
    if mois in [12, 1, 2]:
        return 0
    if mois in [3, 4, 5]:
        return 1
    if mois in [6, 7, 8]:
        return 2
    return 3


def recuperer_observation(station_id, date_heure):
    date_utc = pd.Timestamp(
        date_heure,
        tz="UTC"
    )

    date_sql = date_utc.isoformat(
        sep=" "
    )

    with sqlite3.connect(DATABASE_PATH) as connexion:
        observation = pd.read_sql_query(
            """
            SELECT
                validity_time,
                geo_id_wmo,
                name,
                lat,
                lon,
                dd,
                ff,
                u,
                vv,
                rr1,
                temperature_c,
                point_rosee_c,
                pression_hpa,
                tendance_pression_hpa
            FROM observations_meteo
            WHERE validity_time = ?
              AND geo_id_wmo = ?
            LIMIT 1
            """,
            connexion,
            params=(date_sql, int(station_id))
        )

    return observation


st.set_page_config(
    page_title="Prévision pluie Occitanie",
    page_icon="🌾",
    layout="wide"
)


st.markdown(
    """
    <style>
        .stApp {
            background:
                linear-gradient(
                    135deg,
                    #f5f2e8 0%,
                    #eef3e5 48%,
                    #e5edf1 100%
                );
        }

        .block-container {
            max-width: 1050px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        .header-box {
            padding: 1.7rem 2rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(85, 107, 79, 0.18);
            box-shadow: 0 8px 24px rgba(52, 70, 55, 0.08);
            margin-bottom: 1.5rem;
        }

        .main-title {
            font-size: 2.5rem;
            font-weight: 750;
            line-height: 1.1;
            margin-bottom: 0.5rem;
            color: #253328;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #526056;
            max-width: 760px;
        }

        .section-box {
            padding: 1.4rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(85, 107, 79, 0.14);
            box-shadow: 0 5px 16px rgba(52, 70, 55, 0.06);
            margin-bottom: 1.2rem;
        }

        .info-box {
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.80);
            border-left: 5px solid #596f52;
            color: #39453c;
            margin-top: 1.4rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            min-height: 3rem;
            font-size: 1rem;
            font-weight: 650;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.94);
            padding: 1rem;
            border-radius: 14px;
            border: 1px solid rgba(85, 107, 79, 0.14);
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        h1, h2, h3 {
            color: #253328;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="header-box">
        <div class="main-title">Prévision du risque de pluie</div>
        <div class="subtitle">
            Estimation à court terme pour les stations SYNOP de la région
            Occitanie, à partir d’une observation météorologique disponible.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


modele, features = charger_modele()
stations = charger_stations()

options_stations = {
    f"{row['name']} — {row['geo_id_wmo']}": int(row["geo_id_wmo"])
    for _, row in stations.iterrows()
}


st.markdown(
    '<div class="section-box">',
    unsafe_allow_html=True
)

st.subheader("Paramètres de prévision")

col_station, col_date = st.columns(2)

with col_station:
    station_libelle = st.selectbox(
        "Station météorologique",
        options=list(options_stations.keys()),
        index=None,
        placeholder="Choisir une station"
    )

with col_date:
    date_heure = st.datetime_input(
        "Date et heure de l’observation",
        value=datetime(2025, 1, 1, 0, 0),
        step=10800
    )

st.caption(
    "Les observations SYNOP sont généralement disponibles toutes les trois heures."
)

lancer_prediction = st.button(
    "Estimer le risque de pluie",
    type="primary",
    use_container_width=True
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


if lancer_prediction:
    if station_libelle is None:
        st.error(
            "Veuillez sélectionner une station météorologique."
        )
        st.stop()

    station_id = options_stations[
        station_libelle
    ]

    observation = recuperer_observation(
        station_id,
        date_heure
    )

    if observation.empty:
        st.error(
            "Aucune observation n’est disponible pour cette date "
            "et cette station. Sélectionnez une observation présente "
            "dans la base SYNOP 2025."
        )
        st.stop()

    observation["validity_time"] = pd.to_datetime(
        observation["validity_time"],
        utc=True
    )

    observation["mois"] = (
        observation["validity_time"].dt.month
    )
    observation["jour"] = (
        observation["validity_time"].dt.day
    )
    observation["heure"] = (
        observation["validity_time"].dt.hour
    )
    observation["jour_semaine"] = (
        observation["validity_time"].dt.dayofweek
    )
    observation["saison"] = (
        observation["mois"].apply(
            determiner_saison
        )
    )

    X_prediction = observation[
        features
    ].copy()

    classe = int(
        modele.predict(
            X_prediction
        )[0]
    )

    probabilite = float(
        modele.predict_proba(
            X_prediction
        )[0, 1]
    )

    if probabilite < 0.30:
        niveau = "Faible"
    elif probabilite < 0.60:
        niveau = "Modéré"
    else:
        niveau = "Élevé"

    st.markdown(
        '<div class="section-box">',
        unsafe_allow_html=True
    )

    st.subheader("Résultat de la prévision")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Probabilité de pluie",
        f"{probabilite * 100:.1f} %"
    )

    col2.metric(
        "Niveau de risque",
        niveau
    )

    col3.metric(
        "Horizon",
        "3 heures"
    )

    if classe == 1:
        st.warning(
            "Le modèle estime qu’un épisode de pluie est probable "
            "dans les trois heures suivantes."
        )
    else:
        st.success(
            "Le modèle estime qu’aucun épisode de pluie n’est probable "
            "dans les trois heures suivantes."
        )

    st.write(
        f"**Station :** {observation['name'].iloc[0]}"
    )
    st.write(
        f"**Date observée :** {date_heure}"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div class="info-box">
        <strong>Périmètre du MVP</strong><br>
        L’application utilise les observations météorologiques SYNOP de 2025,
        stockées dans SQLite, pour estimer le risque de pluie dans les trois
        heures suivantes. Une version industrialisée intégrera des données
        météorologiques en temps réel et des prévisions externes afin
        d’étendre l’horizon de prévision à 24 heures.
    </div>
    """,
    unsafe_allow_html=True
)
