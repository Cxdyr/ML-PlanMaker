import pandas as pd
import pickle
from decision_tree import DecisionTreeModel

# Load the model
dt_model = DecisionTreeModel()
dt_model.load_model('../saved_models/decision_tree_model.pkl')
print("Model loaded successfully.")

# Load the label encoder
with open('../saved_models/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Example: Predicting with new data
new_data = pd.DataFrame({
    'Goal': [1],
    'Legs': [1],
    'Chest': [0],
    'Arms': [1],
    'Back': [0],
    'Full Body': [0]
})

print("Input data:")
print(new_data)

# Make predictions
predictions = dt_model.predict(new_data)
predicted_lifts_encoded = predictions[:, 0].astype(int)
predicted_reps = predictions[:, 1]

# Decode Lifts back to the original string format
predicted_lifts_decoded = label_encoder.inverse_transform(predicted_lifts_encoded)

# Display predictions
for lift, reps in zip(predicted_lifts_decoded, predicted_reps):
    print(f"Predicted Lifts: {lift}, Predicted Reps: {int(reps)}")
