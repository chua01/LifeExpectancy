import streamlit as st
import pandas as pd
import joblib

# Load the models and encoder
lr_model = joblib.load('linear_regression_model.pkl')
rf_model_important = joblib.load('random_forest_model.pkl')
encoder = joblib.load('target_encoder.pkl')

# Define the important features based on the model
important_rf_features = ['AdultMortality', 'BMI', 'under-fivedeaths', 'Polio', 'Totalexpenditure',
                         'Diphtheria', 'HIV/AIDS', 'GDP', 'thinness1-19years', 'Incomecompositionofresources']

# Function to predict life expectancy based on input data
def predict_life_expectancy(model, input_data, important_features=None):
    if important_features:
        input_data = input_data[important_features]
    return model.predict(input_data)

# Streamlit app
st.title("Life Expectancy Prediction")

st.write("""
# Predict Life Expectancy
This app predicts the life expectancy based on various health and economic factors.
""")

# Collect user input features
st.sidebar.header("Input Features")
country = st.sidebar.selectbox('Country', ['Afghanistan', 'CountryName'])  # Example values
year = st.sidebar.slider('Year', 2000, 2015, 2015)
status = st.sidebar.selectbox('Status', ['Developed', 'Developing'])
gdp = st.sidebar.number_input('GDP', min_value=0, value=40000)
schooling = st.sidebar.number_input('Schooling', min_value=0, max_value=20, value=15)
income_composition = st.sidebar.number_input('Income Composition of Resources', min_value=0.0, max_value=1.0, value=0.8)

# Create a dictionary with user inputs
example_data = {
    'Country': [country],
    'Year': [year],
    'Status': [status],
    'GDP': [gdp],
    'Schooling': [schooling],
    'Incomecompositionofresources': [income_composition]
}

# Fill in missing columns with default values
all_features = important_rf_features + ['Year', 'Country', 'Status']
for col in all_features:
    if col not in example_data:
        example_data[col] = [0]  # Default value, replace with mean or mode as appropriate

input_data = pd.DataFrame(example_data)

# Encode categorical features
input_data[['Country', 'Status']] = encoder.transform(input_data[['Country', 'Status']])

# Predict using the chosen model
# Assuming RandomForest is chosen based on previous evaluation
prediction = predict_life_expectancy(rf_model_important, input_data, important_rf_features)
chosen_model = "Random Forest (Important Features)"

st.write(f"The chosen model for deployment is: {chosen_model}")
st.write(f"Predicted Life Expectancy: {prediction[0]:.2f}")
