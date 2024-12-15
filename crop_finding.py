from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the dataset
recipe_df = pd.read_csv('Crop_recommendation.csv')

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(recipe_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=2, metric='euclidean')
knn.fit(X_numerical)

# Function to Recommend Crops
def recommend_crops(input_features):
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_df = pd.DataFrame([input_features], columns=feature_names)
    input_features_scaled = scaler.transform(input_df)
    distances, indices = knn.kneighbors(input_features_scaled)
    crops = recipe_df.iloc[indices[0]]['label'].tolist()
    return crops

# API Route
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    input_features = data.get('features')  # Expecting a JSON object with 'features' key
    if not input_features or len(input_features) != 7:
        return jsonify({'error': 'Invalid input. Please provide 7 feature values.'}), 400
    recommended_crops = recommend_crops(input_features)
    return jsonify({'recommended_crops': recommended_crops})

if __name__ == '__main__':
    app.run(debug=True)
