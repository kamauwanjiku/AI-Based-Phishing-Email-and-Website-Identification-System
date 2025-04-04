from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained models
url_model = joblib.load('url_model.pkl')
email_model = joblib.load('email_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/check-url', methods=['POST'])
def check_url():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Extract features
    features = extract_url_features(url)
    feature_vector = [list(features.values())]
    
    # Make prediction
    prediction = url_model.predict_proba(feature_vector)[0, 1]
    is_phishing = prediction > 0.5
    
    return jsonify({
        'url': url,
        'is_phishing': bool(is_phishing),
        'confidence': float(prediction),
        'features': features
    })

@app.route('/api/check-email', methods=['POST'])
def check_email():
    data = request.json
    email_content = data.get('email')
    
    if not email_content:
        return jsonify({'error': 'No email content provided'}), 400
    
    # Extract features
    features = extract_email_features(email_content)
    feature_vector = [list(features.values())]
    
    # Make prediction
    prediction = email_model.predict_proba(feature_vector)[0, 1]
    is_phishing = prediction > 0.5
    
    return jsonify({
        'is_phishing': bool(is_phishing),
        'confidence': float(prediction),
        'features': features
    })

if __name__ == '__main__':
    app.run(debug=True)