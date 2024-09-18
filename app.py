from flask import Flask, render_template, request
import joblib

# Load the model and vectorizer
model = joblib.load(r'C:\Users\yuvra\Downloads\sentiment_model.pkl')
vectorizer = joblib.load(r'C:\Users\yuvra\Downloads\tfidf_vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    user_input = request.form['comment']
    
    # Transform the input text using the vectorizer
    input_vector = vectorizer.transform([user_input])
    
    # Make a prediction
    prediction = model.predict(input_vector)
    
    # Convert prediction to readable form
    sentiment = prediction[0]

    # Render the result
    return render_template('index.html', prediction_text=f'This comment is {sentiment}.')

if __name__ == '__main__':
    app.run(debug=True)
