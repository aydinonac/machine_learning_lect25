from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# Load data
data = np.loadtxt("reduced.csv", delimiter=",")
labels = np.random.randint(0, 2, size=data.shape[0])  # Dummy labels

# Train a logistic regression model
model = LogisticRegression()
model.fit(data.reshape(-1, 1), labels)

# Save the model
joblib.dump(model, "classifier.pkl")
