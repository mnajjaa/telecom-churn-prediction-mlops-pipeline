import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from scipy.stats import zscore

def prepare_data(train_path, test_path):
    """
    Charge et prépare les données pour l'entraînement et le test.
    
    Parameters:
    train_path (str): Chemin du fichier CSV d'entraînement.
    test_path (str): Chemin du fichier CSV de test.
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Fusionner pour assurer une transformation identique
    data = pd.concat([train_data, test_data], ignore_index=True)
    
    # Supprimer les colonnes inutiles
    columns_to_drop = ['State', 'Area code', 'Total day minutes',
                       'Total eve minutes', 'Total night minutes', 'Total intl minutes']
    data = data.drop(columns=columns_to_drop, errors='ignore')
    
    # Encodage des variables catégorielles
    label_encoder = LabelEncoder()
    for col in ['International plan', 'Voice mail plan', 'Churn']:
        data[col] = label_encoder.fit_transform(data[col])
    
    # Normalisation des données numériques
    numerical_columns = ['Account length', 'Number vmail messages', 'Total day calls',
                         'Total day charge', 'Total eve calls', 'Total eve charge',
                         'Total night calls', 'Total night charge', 'Total intl calls',
                         'Total intl charge', 'Customer service calls']
    
    data[numerical_columns] = MinMaxScaler().fit_transform(data[numerical_columns])
    
    # Séparer les données en ensembles d'entraînement et de test
    train_data = data.iloc[:len(train_data)]
    test_data = data.iloc[len(train_data):]
    
    X_train, y_train = train_data.drop('Churn', axis=1), train_data['Churn']
    X_test, y_test = test_data.drop('Churn', axis=1), test_data['Churn']
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Entraîne un modèle RandomForestClassifier.
    
    Parameters:
    X_train (pd.DataFrame): Données d'entraînement.
    y_train (pd.Series): Labels d'entraînement.
    
    Returns:
    RandomForestClassifier: Modèle entraîné.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur l'ensemble de test.
    
    Parameters:
    model (RandomForestClassifier): Modèle entraîné.
    X_test (pd.DataFrame): Données de test.
    y_test (pd.Series): Labels de test.
    
    Returns:
    tuple: (accuracy, precision, recall, f1_score)
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def save_model(model, filename):
    """
    Sauvegarde le modèle entraîné.
    
    Parameters:
    model (RandomForestClassifier): Modèle entraîné.
    filename (str): Nom du fichier de sauvegarde.
    """
    joblib.dump(model, filename)

def load_model(filename):
    """
    Charge un modèle sauvegardé.
    
    Parameters:
    filename (str): Nom du fichier du modèle sauvegardé.
    
    Returns:
    RandomForestClassifier: Modèle chargé.
    """
    return joblib.load(filename)
