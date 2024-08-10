# Telecom Customer Churn Prediction

This project focuses on predicting customer churn in the telecom industry using a machine learning model. The model is built using the Random Forest algorithm and trained on a Kaggle dataset. The project also includes a web application developed using Flask, allowing users to input customer data and get predictions on whether a customer is likely to churn or not.

## Table of Contents

- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Web Application](#web-application)
- [How to Run](#how-to-run)
- [Results](#results)
- [Contributing](#contributing)

## Dataset

The dataset used in this project is sourced from Kaggle and contains customer data from a telecom company. It includes various features like customer tenure, monthly charges, total charges, and several categorical features like gender, contract type, and payment method.

- **Filename:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Size:** ~10,000 rows and 21 columns

## Data Preprocessing

### Steps Taken:
1. **Handling Missing Values:**
   - Converted the `TotalCharges` column to numeric and handled missing values by removing rows with missing data.
   
2. **Feature Engineering:**
   - Grouped the `tenure` column into bins representing different time periods.
   - Dropped irrelevant columns like `customerID` and `tenure`.

3. **Encoding Categorical Variables:**
   - Converted categorical variables into dummy variables using `pd.get_dummies()`.

## Exploratory Data Analysis (EDA)

The EDA involved visualizing the distribution of features and their relationship with the target variable (`Churn`):

- **Target Variable Distribution:** Analyzed the distribution of churned vs. non-churned customers.
- **Correlation Analysis:** Heatmaps and bar charts were used to identify correlations between features and churn.
- **Univariate and Bivariate Analysis:** Plots were generated to explore the relationship between individual features and churn.

## Model Building

### Steps:
1. **Train-Test Split:** Split the data into training and testing sets with an 80-20 split.
2. **Model Selection:** Used a Random Forest Classifier for its ability to handle large datasets and provide feature importance.
3. **Handling Imbalanced Data:** Used SMOTEENN to handle the imbalanced nature of the dataset.

### Model Parameters:
- **n_estimators:** 100
- **max_depth:** 6
- **min_samples_leaf:** 8
- **random_state:** 100

## Model Evaluation

The model was evaluated using metrics such as:

- **Accuracy**
- **Recall**
- **Precision**
- **Confusion Matrix**

Results were compared before and after applying SMOTEENN to balance the dataset.

## Web Application

A Flask-based web application was created to allow users to input customer data and receive predictions on whether a customer is likely to churn. The application loads the trained model and provides predictions along with the confidence level.

### Files:
- **app.py:** The main Flask application file.
- **home.html:** The frontend HTML file for the web interface.

## How to Run

### Prerequisites:
- Python 3.x
- Required Python packages: `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `imblearn`, `flask`, `joblib`

### Steps:
1. Clone the repository:
   ```shell
   git clone https://github.com/yourusername/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```
2. Install the required packages:
   ```shell
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```shell
   python app.py
   ```
4. Open your web browser and navigate to `http://127.0.0.1:5000/`.

### Results:
The model achieved a satisfactory level of accuracy and recall, especially after handling the imbalanced dataset using SMOTEENN. The web application provides an easy-to-use interface for predicting customer churn with a clear confidence score.

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bug or feature request.

