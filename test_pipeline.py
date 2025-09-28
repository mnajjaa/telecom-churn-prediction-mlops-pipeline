import pytest
import pandas as pd
import numpy as np
from model_pipeline1 import prepare_data, train_model, evaluate_model, save_model, load_model

def test_prepare_data():
    """Teste la fonction prepare_data pour s'assurer qu'elle charge et prétraite les données correctement."""
    X_train, X_test, y_train, y_test = prepare_data("churn-bigml-80.csv", "churn-bigml-20.csv")
    assert X_train is not None and X_test is not None
    assert y_train is not None and y_test is not None
    assert len(X_train) > 0 and len(X_test) > 0

def test_train_model():
    """Teste l'entraînement du modèle pour s'assurer qu'il s'exécute sans erreurs."""
    X_train = pd.DataFrame(np.random.rand(10, 5))
    y_train = pd.Series(np.random.randint(0, 2, 10))
    model = train_model(X_train, y_train)
    assert model is not None
    assert hasattr(model, "predict")

def test_evaluate_model():
    """Teste l'évaluation du modèle pour vérifier les métriques."""
    X_train = pd.DataFrame(np.random.rand(10, 5))
    y_train = pd.Series(np.random.randint(0, 2, 10))
    model = train_model(X_train, y_train)
    X_test = pd.DataFrame(np.random.rand(5, 5))
    y_test = pd.Series(np.random.randint(0, 2, 5))
    acc, prec, rec, f1 = evaluate_model(model, X_test, y_test)
    assert 0 <= acc <= 1
    assert 0 <= prec <= 1
    assert 0 <= rec <= 1
    assert 0 <= f1 <= 1

def test_save_and_load_model():
    """Teste la sauvegarde et le chargement du modèle."""
    X_train = pd.DataFrame(np.random.rand(10, 5))
    y_train = pd.Series(np.random.randint(0, 2, 10))
    model = train_model(X_train, y_train)
    save_model(model, "test_model.pkl")
    loaded_model = load_model("test_model.pkl")
    assert hasattr(loaded_model, "predict")
