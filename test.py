import joblib
import numpy as np

# Load the pre-trained machine learning model (replace 'model.pkl' with your model file)
model = joblib.load('model.pkl')

# Sample input data for testing (replace with your own data)
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example input features

# Make predictions using the loaded model
predicted_class = model.predict(sample_data)
predicted_class_probabilities = model.predict_proba(sample_data)

# Print the prediction result
print("Predicted class:", predicted_class[0])
print("Predicted class probabilities:", predicted_class_probabilities[0])

