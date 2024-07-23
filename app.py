from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained models
model_severity = pickle.load(open('severity.pkl', 'rb'))
model_response = pickle.load(open('response.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/references')
def references():
    return render_template('Refrances.html')

@app.route('/effects')
def effects():
    return render_template('Effects.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    features = {
        'Species': [request.form['species']],
        'Organism': [request.form['organism']],
        'UV': [request.form['uv']],
        'Exposure intensity (in J, KJ, W, KW)': [request.form['intensity']],
        'Exposure time (in seconds)': [float(request.form['time'])],
        'Organelle': [request.form['organelle']],
        'Metabolites names': [request.form['metabolites']],
        'Proteins names': [request.form['proteins']],
        'Genes names': [request.form['genes']],
        'Studied tissue': [request.form['tissue']]
    }
    
    # Convert features to DataFrame
    final_features = pd.DataFrame(features)

    # Make predictions
    severity_prediction = model_severity.predict(final_features)
    response_prediction = model_response.predict(final_features)

    return render_template('index.html', 
                           severity_prediction=severity_prediction[0],
                           response_prediction=response_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
