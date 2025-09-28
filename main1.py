import argparse
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from model_pipeline1 import prepare_data, train_model, evaluate_model, save_model, load_model
from elasticsearch import Elasticsearch

# Connexion à Elasticsearch
es = Elasticsearch("http://elasticsearch:9200")

def log_to_elasticsearch(metrics):
    """ Envoie les métriques MLflow vers Elasticsearch """
    es.index(index="mlflow-metrics", body=metrics)

def main():
    """
    Programme principal pour exécuter l'entraînement et l'évaluation du modèle.
    """
    parser = argparse.ArgumentParser(description="Train and Evaluate Churn Prediction Model")
    parser.add_argument("--train_path", type=str, help="Path to training dataset")
    parser.add_argument("--test_path", type=str, help="Path to testing dataset")
    parser.add_argument("--save_model", type=str, default='churn_model.pkl', help="Path to save trained model")
    parser.add_argument("--load_model", type=str, help="Path to load existing model for evaluation")
    
    args = parser.parse_args()
    
    # Définir l'expérience MLflow
    mlflow.set_tracking_uri("http://0.0.0.0:8000")  # Utiliser MLflow sans SQLite
    mlflow.set_experiment("Churn Prediction")
    
    with mlflow.start_run():
        mlflow.log_param("train_path", args.train_path)
        mlflow.log_param("test_path", args.test_path)

        # Vérification si nous effectuons un entraînement ou une évaluation
        if args.load_model:
            model = load_model(args.load_model)
            print(f"📂 Modèle chargé depuis {args.load_model}")
            
            if args.test_path:
                _, X_test, _, y_test = prepare_data(args.test_path, args.test_path)
                accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
                
                print(f"📊 Accuracy: {accuracy}")
                print(f"🎯 Precision: {precision}")
                print(f"🔁 Recall: {recall}")
                print(f"🏆 F1 Score: {f1}")

                # Enregistrer les métriques dans MLflow et Elasticsearch
                metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
                mlflow.log_metrics(metrics)
                log_to_elasticsearch(metrics)
            else:
                raise ValueError("Vous devez fournir `--test_path` pour évaluer un modèle chargé.")
        elif args.train_path and args.test_path:
            X_train, X_test, y_train, y_test = prepare_data(args.train_path, args.test_path)
            model = train_model(X_train, y_train)
            save_model(model, args.save_model)
            print(f"💾 Modèle enregistré sous {args.save_model}")
            
            accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
            print(f"📊 Accuracy: {accuracy}")
            print(f"🎯 Precision: {precision}")
            print(f"🔁 Recall: {recall}")
            print(f"🏆 F1 Score: {f1}")

            # Enregistrer les métriques dans MLflow et Elasticsearch
            metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
            mlflow.log_metrics(metrics)
            log_to_elasticsearch(metrics)

            # Sauvegarde du modèle dans MLflow
            mlflow.sklearn.log_model(model, "model_churn")
        else:
            raise ValueError("Vous devez spécifier `--train_path` et `--test_path` pour entraîner ou `--load_model` avec `--test_path` pour évaluer.")

if __name__ == "__main__":
    main()
