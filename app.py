
from flask import Flask, jsonify, request
import joblib
import pandas as pd
import sqlite3

app = Flask(__name__)

MODELE_PATH = "modele_pluie.joblib"
FEATURES_PATH = "features_modele.joblib"
DATABASE_PATH = "goutte_eau.db"

modele = joblib.load(MODELE_PATH)
features = joblib.load(FEATURES_PATH)


def determiner_saison(mois):
    if mois in [12, 1, 2]:
        return 0
    elif mois in [3, 4, 5]:
        return 1
    elif mois in [6, 7, 8]:
        return 2
    return 3


def recuperer_observation(date_sql, station_id):
    connexion = sqlite3.connect(DATABASE_PATH)

    requete = """
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
    """

    observation = pd.read_sql_query(
        requete,
        connexion,
        params=(date_sql, station_id)
    )

    connexion.close()
    return observation


@app.route("/", methods=["GET"])
def accueil():
    return jsonify({
        "nom": "API Goutte d'Eau",
        "description": "Estimation du risque de pluie à trois heures",
        "routes": {
            "sante": "GET /health",
            "stations": "GET /stations",
            "prediction": "POST /predict"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "modele": "Régression Logistique"
    })


@app.route("/stations", methods=["GET"])
def stations():
    connexion = sqlite3.connect(DATABASE_PATH)

    resultat = pd.read_sql_query(
        """
        SELECT DISTINCT
            geo_id_wmo AS station_id,
            name AS station
        FROM observations_meteo
        ORDER BY geo_id_wmo
        """,
        connexion
    )

    connexion.close()

    return jsonify(
        resultat.to_dict(orient="records")
    )


@app.route("/predict", methods=["POST"])
def predict():
    contenu = request.get_json(silent=True)

    if contenu is None:
        return jsonify({
            "erreur": "Le corps doit être au format JSON."
        }), 400

    date_demandee = contenu.get("date")
    station_id = contenu.get("station_id")

    if not date_demandee or station_id is None:
        return jsonify({
            "erreur": "Les champs date et station_id sont obligatoires."
        }), 400

    try:
        station_id = int(station_id)
        date_convertie = pd.to_datetime(
            date_demandee,
            utc=True
        )
    except (TypeError, ValueError):
        return jsonify({
            "erreur": "Date ou identifiant de station invalide."
        }), 400

    date_sql = date_convertie.isoformat(sep=" ")

    observation = recuperer_observation(
        date_sql,
        station_id
    )

    if observation.empty:
        return jsonify({
            "erreur": "Aucune observation trouvée pour cette date et cette station."
        }), 404

    observation["validity_time"] = pd.to_datetime(
        observation["validity_time"],
        utc=True
    )

    observation["mois"] = observation["validity_time"].dt.month
    observation["jour"] = observation["validity_time"].dt.day
    observation["heure"] = observation["validity_time"].dt.hour
    observation["jour_semaine"] = observation["validity_time"].dt.dayofweek
    observation["saison"] = observation["mois"].apply(
        determiner_saison
    )

    X_prediction = observation[features].copy()

    classe = int(
        modele.predict(X_prediction)[0]
    )

    probabilite = float(
        modele.predict_proba(X_prediction)[0, 1]
    )

    if probabilite < 0.30:
        niveau = "faible"
    elif probabilite < 0.60:
        niveau = "modéré"
    else:
        niveau = "élevé"

    return jsonify({
        "date_observation": str(date_demandee),
        "station_id": station_id,
        "station": observation["name"].iloc[0],
        "horizon_prediction": "3 heures",
        "classe_predite": classe,
        "pluie_predite": bool(classe),
        "probabilite_pluie": round(probabilite, 4),
        "probabilite_pluie_pourcentage": round(
            probabilite * 100,
            2
        ),
        "niveau_risque": niveau
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
