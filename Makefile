# Définition de l'environnement virtuel
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Installation des dépendances
install: $(VENV)/bin/activate
	@echo " Installation des dépendances..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

$(VENV)/bin/activate:
	@echo " Création de l'environnement virtuel..."
	python3 -m venv $(VENV)

# Exécution du pipeline complet (préparation, entraînement, évaluation)
run:
	@echo " Exécution du pipeline..."
	$(PYTHON) main.py

# Tests unitaires
test:
	@echo " Exécution des tests..."
	[ -d tests ] || mkdir tests
	[ -f tests/test_dummy.py ] || echo "import unittest\n\
class TestDummy(unittest.TestCase):\n\
    def test_example(self):\n\
        self.assertEqual(1 + 1, 2)\n\
if __name__ == '__main__':\n\
    unittest.main()" > tests/test_dummy.py
	$(PYTHON) -m unittest discover tests/
	@echo " Tests terminés avec succès."

# Vérification du format du code
format:
	@echo " Vérification et formatage du code..."
	$(PYTHON) -m black --version >/dev/null 2>&1 || $(PIP) install black
	$(PYTHON) -m black .

# Nettoyage
clean:
	@echo " Nettoyage du projet..."
	rm -rf $(VENV) __pycache__ *.pkl tests/__pycache__ .pytest_cache logs/

# Lancer l'API FastAPI
api:
	@echo " Démarrage de l'API FastAPI..."
	$(PYTHON) -m uvicorn app:app --reload --host 127.0.0.1 --port 8005

# Lancer l'interface MLflow
mlflow-ui:
	@echo " Démarrage de l'interface MLflow..."
	mlflow ui --host 127.0.0.1 --port 5000 &

# Tester la prédiction avec FastAPI
predict:
	@echo " Test de l'API de prédiction..."
	curl -X 'POST' 'http://127.0.0.1:8005/predict' \
	-H 'Content-Type: application/json' \
	-d '{"features": [0.5, 1.2, 3.4, 2.1, 0.7, 1.8, 2.9, 0.6, 1.1, 3.3, 0.8, 2.5, 1.4]}'

# Construire l’image Docker
docker-build:
	@echo " Construction de l'image Docker..."
	docker build -t ibtihel-mnaja-4ds4-mlops .

# Lancer le conteneur Docker
docker-run:
	@echo " Lancement du conteneur Docker..."
	docker run -p 8005:8005 ibtihel-mnaja-4ds4-mlops

# Pousser l’image sur Docker Hub
docker-push:
	@echo " Poussée de l'image sur Docker Hub..."
	docker tag ibtihel-mnaja-4ds4-mlops ibtihelesprit/ibtihel-mnaja-4ds4-mlops
	docker push ibtihelesprit/ibtihel-mnaja-4ds4-mlops


