import pickle
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)
with open('sepsis_survival.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/output', methods=['POST'])
def output():
    if request.method == 'POST':
        try:
            age_years = int(request.form['age'])
            sex_0male_1female = int(request.form['gender'])
            episode_number = float(request.form['episode_number'])

            features = [[age_years, sex_0male_1female, episode_number]]
            print(f"Input features: {features}")

            # Create a DataFrame with appropriate feature names
            feature_names = ['age_years', 'sex_0male_1female', 'episode_number']
            features_df = pd.DataFrame(features, columns=feature_names)

            # Print DataFrame to debug input
            print(f"Features DataFrame:\n{features_df}")

            # Predict the outcome
            prediction = model.predict(features_df)[0]
            print(f"Raw prediction: {prediction}")

            # Handle prediction output
            if prediction == 0:
                result = "Dead"
            elif prediction == 1:
                result = "Alive"
            else:
                result = "Unknown"

            print(f"Result: {result}")
            return render_template("output.html", prediction=result)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template("output.html", prediction="Error during prediction")

if __name__ == '__main__':
    app.run(debug=True)