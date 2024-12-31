from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the spam/ham model
with open('spam_ham_model.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

with open('Sentiment.pkl', 'rb') as model_file1:
    model2 = joblib.load(model_file1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.json
        message = data.get('message', '')

        if not message:
            return jsonify({"error": "Message field is required"}), 400

        # Use the model to make a prediction
        prediction = model.predict([message])
        prediction2 = model2.predict([message])
        # Return the prediction as JSON
        return jsonify({"message": message, "reply": prediction[0], "reply2":prediction2[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080, debug=True)
