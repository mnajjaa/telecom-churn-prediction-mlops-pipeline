FROM python:3.8

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt ./

# Install dependencies
RUN apt-get update && apt-get install -y libpq-dev && \
    pip install --no-cache-dir -r requirements.txt psycopg2-binary elasticsearch mlflow

# Copy application files
COPY . .

# Expose necessary ports for Flask, MLflow, PostgreSQL, Elasticsearch, and Kibana
EXPOSE 8082 8000 5432 9200 5601

# Define environment variables
ENV MODEL_PATH="churn_model.pkl"
ENV DATABASE_URL="postgresql://admin:admin@db:5432/predictions_db"

# Run both Flask and MLflow UI
CMD mlflow ui --host 0.0.0.0 --port 8000 & python app_flask.py