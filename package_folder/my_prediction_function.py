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
from package_folder.preprocessor_light import * #load_loan_data

def creating_full_dataframe_from_inputs(loan_limit, income, age):
    # Get the path to the Loan_Default.csv file (raw data))
    ROOT_PATH = pathlib.Path().resolve().parent # Get the parent directory of the current working directory
    raw_data_path = os.path.join(ROOT_PATH, 'raw_data', 'Loan_Default.csv')

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
    preprocessor = process_data()
    X_user_processed = preprocessor.transform(X_user).drop(columns="Status")

    # Load the model from the pretrain model pickle file
    ROOT_PATH = pathlib.Path().resolve().parent # Get the parent directory of the current working directory
    model_path = os.path.join(ROOT_PATH, 'models', 'mvp_model.pkl')
    print(f"Path of the model.pkl:\n{model_path}\n")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Predict
    prediction = model.predict(X_user_processed)
    print(f"Prediction: {prediction[0]}")

    print("✅ Prediction done succesfully")

    return prediction

# Run the processing pipeline
if __name__ == "__main__":
    my_prediction_function("cf", 5760.0, "45-54")
