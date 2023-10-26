# Import necessary libraries
import pandas as pd
from flask import Flask, request, render_template
import joblib

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load the trained model
model = joblib.load("final.pkl")

@app.route("/")
def loadPage():
    return render_template('home.html', output1="", output2="", query1="", query2="", query3="", query4="", query5="", query6="", query7="", query8="", query9="", query10="", query11="", query12="", query13="", query14="", query15="", query16="", query17="", query18="", query19="")

@app.route("/", methods=['POST'])
def predict():
    # Receive input data from the web form
    input_data = {
        'SeniorCitizen': int(request.form['query1']),
        'MonthlyCharges': float(request.form['query2']),
        'TotalCharges': float(request.form['query3']),
        'gender': request.form['query4'],
        'Partner': request.form['query5'],
        'Dependents': request.form['query6'],
        'PhoneService': request.form['query7'],
        'MultipleLines': request.form['query8'],
        'InternetService': request.form['query9'],
        'OnlineSecurity': request.form['query10'],
        'OnlineBackup': request.form['query11'],
        'DeviceProtection': request.form['query12'],
        'TechSupport': request.form['query13'],
        'StreamingTV': request.form['query14'],
        'StreamingMovies': request.form['query15'],
        'Contract': request.form['query16'],
        'PaperlessBilling': request.form['query17'],
        'PaymentMethod': request.form['query18'],
        'tenure': int(request.form['query19'])
    }

    # Convert the input data into a DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # Make predictions using the loaded model
    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[:, 1]

    # Prepare output messages
    if predictions[0] == 1:
        output1 = "This customer is likely to be churned!!"
    else:
        output1 = "This customer is likely to continue!!"
    
    output2 = "Confidence: {:.2f}%".format(probabilities * 100)

    return render_template('home.html', output1=output1, output2=output2, **input_data)

# if __name__ == "__main":
#     app.run(debug=True)
app.run()