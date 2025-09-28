import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from model_pipeline import prepare_data, train_model, evaluate_model, save_model

if __name__ == "__main__":
    train_path = 'churn-bigml-80.csv'
    test_path = 'churn-bigml-20.csv'
    
    mlflow.set_experiment("Churn Prediction")

    with mlflow.start_run():
        mlflow.log_param("train_data", train_path)
        mlflow.log_param("test_data", test_path)

        # Préparation des données
        X_train, X_test, y_train, y_test = prepare_data(train_path, test_path)

        # Entraînement du modèle
        model = train_model(X_train, y_train)

        # Évaluation du modèle
        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # ✅ Correction : Conversion en NumPy avec `.iloc[0].to_numpy()`
        input_example = np.expand_dims(X_train.iloc[0].to_numpy(), axis=0)

        # Enregistrer le modèle avec l'exemple d'entrée
        mlflow.sklearn.log_model(model, "churn_model", input_example=input_example)

        print(" Modèle et métriques enregistrés avec MLflow !")
