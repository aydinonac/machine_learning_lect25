import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import joblib

# Load the trained model
with open("classifier.pkl", "rb") as model_file:
    model = joblib.load(model_file)

# Load the reduced data
data = np.loadtxt("reduced.csv", delimiter=",")

# Generate random labels for demonstration purposes
labels = np.random.randint(0, 2, size=data.shape[0])

# Make predictions using the model
predictions = model.predict(data.reshape(-1, 1))

# Calculate metrics
accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)
roc_auc = roc_auc_score(labels, predictions)

# Write metrics to scores.json
metrics = {"accuracy": accuracy, "precision": precision, "roc_auc": roc_auc}
with open("scores.json", "w") as scores_file:
    json.dump(metrics, scores_file)
