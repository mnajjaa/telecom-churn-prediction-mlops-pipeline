from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List

# Définition de l’API FastAPI
app = FastAPI(title="API de Prédiction du Churn", version="1.1")

# Chargement sécurisé du modèle
MODEL_PATH = "churn_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False

# Définition du format d’entrée pour les prédictions
class PredictionInput(BaseModel):
    features: List[float]

@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Effectue une prédiction à partir des features envoyées par l’utilisateur.
    """
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Modèle non chargé. Veuillez l'entraîner et le sauvegarder.")

    # Vérification de l’entrée utilisateur
    if not isinstance(data.features, list) or len(data.features) == 0:
        raise HTTPException(status_code=400, detail="Les features doivent être une liste de valeurs numériques.")

    try:
        features_array = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features_array)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {str(e)}")

@app.get("/healthcheck")
async def healthcheck():
    """
    Vérifie l'état de l'API et du modèle.
    """
    if model_loaded:
        return {"status": "ok", "message": "API fonctionnelle et modèle chargé."}
    else:
        return {"status": "error", "message": "Modèle non chargé. Vérifiez son chemin ou réentraînez-le."}

@app.post("/retrain")
async def retrain():
    """
    Réentraîne le modèle avec les données d'entraînement et le met à jour.
    """
    global model  # Permet de mettre à jour le modèle en mémoire
    
    try:
        # Charger et préparer les données
        X_train, X_test, y_train, y_test = prepare_data("churn-bigml-80.csv", "churn-bigml-20.csv")
        
        # Réentraîner le modèle
        new_model = train_model(X_train, y_train)
        
        # Sauvegarder le modèle
        save_model(new_model, MODEL_PATH)
        
        # Charger le nouveau modèle en mémoire
        model = joblib.load(MODEL_PATH)
        
        return {"status": "ok", "message": "Modèle réentraîné et mis à jour avec succès."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du réentraînement : {str(e)}")
