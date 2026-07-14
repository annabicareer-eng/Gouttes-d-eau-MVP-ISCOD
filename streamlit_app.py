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
    page_title="Projet Goutte d’Eau",
    page_icon="💧",
    layout="centered"
)

st.title("💧 Projet Goutte d’Eau")

st.write(
    "Estimation du risque de pluie dans les trois heures suivantes "
    "à partir d’une observation météorologique disponible."
)

modele, features = charger_modele()
stations = charger_stations()

options_stations = {
    f"{row['name']} — {row['geo_id_wmo']}": int(row["geo_id_wmo"])
    for _, row in stations.iterrows()
}

station_libelle = st.selectbox(
    "Station météorologique",
    options=list(options_stations.keys()),
    index=None,
    placeholder="Choisir une station"
)

date_heure = st.datetime_input(
    "Date et heure de l’observation",
    value=datetime(2025, 1, 1, 0, 0),
    step=10800
)

st.caption(
    "Les observations SYNOP utilisées sont généralement espacées de trois heures."
)

if st.button(
    "Estimer le risque de pluie",
    type="primary",
    use_container_width=True
):
    if station_libelle is None:
        st.error("Veuillez sélectionner une station.")
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
            "Aucune observation trouvée pour cette date et cette station. "
            "Le MVP nécessite une observation déjà enregistrée dans la base."
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

    st.subheader("Résultat")

    col1, col2 = st.columns(2)

    col1.metric(
        "Probabilité de pluie",
        f"{probabilite * 100:.2f} %"
    )

    col2.metric(
        "Niveau de risque",
        niveau
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
    st.write(
        "**Horizon :** trois heures"
    )

st.divider()

st.info(
    "Limite du MVP : l’application utilise des observations déjà présentes "
    "dans SQLite. Une version industrialisée intégrerait des données météo "
    "en temps réel ou des prévisions externes."
)
