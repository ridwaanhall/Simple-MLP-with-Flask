import numpy as np
from flask import Flask, jsonify, request
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)

# Define a simple MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000)

# Example training data
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 0])

# Train the MLP model
mlp_model.fit(X_train, y_train)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'This is a simple MLP classifier API by ridwaanhall'})
    #'This is a simple MLP classifier API by ridwaanhall'

@app.route('/predict', methods=['POST'])
def prediction():
    data = request.json
    if data is not None:
        try:
            features = data['features']
            features_array = np.array(features).reshape(1, -1)
            prediction = mlp_model.predict(features_array)
            return jsonify({"prediction": int(prediction[0])})
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "No JSON data received"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
