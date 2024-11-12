# def my_prediction_function(age,income,loan_limit):
#     if int(age)>20 and int(income)>20000 and int(loan_limit)<int(income*10):
#         return f"your application for a loan of {loan_limit} EUR is approved"
#     else:
#         return f"your application for a loan of {loan_limit} EUR is not approved"

# print(my_prediction_function(21,30000,500000))

# General
import os
import pathlib
import pickle

# Analysis
import pandas as pd

# Machine learning
from sklearn.linear_model import LogisticRegression

# Import preprocess_light functions
from package_folder.preprocessor_for_regression import *

def creating_full_dataframe_from_inputs(loan_limit, income, age):
    # Get the path to the Loan_Default.csv file (raw data))
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    raw_data_path = os.path.join(ROOT_PATH, 'raw_data', 'Loan_Default.csv')

    # Printing raw_data_path
    print('Raw_data_path: ', raw_data_path)

    # Convert the Loan_Default.file into a DataFrame
    data_raw = pd.read_csv(raw_data_path)

    # Creating the input Dataframe
    X_user = pd.DataFrame(data_raw.iloc[0, :]).transpose()#.drop(columns="Status")
    X_user["loan_limit"] = loan_limit
    X_user["income"] = income
    X_user["age"] = age

    print("✅ Input dataframe created successfully")

    return X_user


def my_prediction_function(loan_limit, income, age):
    # Create the input dataframe
    X_user = creating_full_dataframe_from_inputs(loan_limit, income, age)

    # Load the preprocessor and transform the input dataframe
    full_pipeline = process_data()
    X_user_processed = full_pipeline.transform(X_user)
    columns=[f"PC{i}" for i in range(1,25)]+['rate_of_interest']
    X_user_processed = pd.DataFrame(X_user_processed,columns=columns).drop(columns='rate_of_interest')
    print(f"X_user_processed: {X_user_processed}")

    # Load the model from the pretrain model pickle file
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    linear_regression_model_path = os.path.join(ROOT_PATH, 'models', 'linear_regression')
    print(f"Path of the linear_regression:\n{linear_regression_model_path}\n")
    with open(linear_regression_model_path, 'rb') as file:
        model = pickle.load(file)

    # Predict
    print(f"X_user_processed.shape: {X_user_processed.shape}")
    prediction = model.predict(X_user_processed)
    print(f"Interest rate: {prediction}")

    print("✅ Prediction done succesfully")

    return prediction

# Run the processing pipeline
if __name__ == "__main__":
    my_prediction_function("cf", 5760.0, "45-54")
