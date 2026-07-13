# Projet Goutte d'Eau

## Présentation

Ce projet a été réalisé dans le cadre du Projet Goutte d'Eau.

L'objectif est de développer un MVP permettant d'estimer le risque de pluie dans les trois heures suivantes à partir des observations météorologiques SYNOP de Météo-France.

## Technologies utilisées

- Python
- Pandas
- SQLite
- Scikit-learn
- Flask
- Joblib

## Structure du projet

- app.py : API Flask
- goutte_eau.db : base de données SQLite
- modele_pluie.joblib : modèle entraîné
- features_modele.joblib : variables utilisées par le modèle
- goutte_eau_mvp.ipynb : notebook de développement
- requirements.txt : dépendances Python

## Modèle retenu

Le modèle final est une Régression Logistique.

Les performances obtenues sont :

- Accuracy : 0.722
- Precision : 0.245
- Recall : 0.765
- F1-score : 0.371

## API

Routes disponibles :

- GET /health
- GET /stations
- POST /predict

## Interface

- Gradio => Accesible via : https://56cc3427beefefcff7.gradio.live (si vous testez le code, le lien est généré à la fin)

## Auteur

Chemseddine Annabi
