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

# Import preprocess_light functions
from package_folder.create_df_from_inputs import * #load_loan_data

def my_prediction_function(loan_limit=None, Gender=None, open_credit=None,
                     business_or_commercial=None, loan_amount=None,
                     term=None, interest_only=None, lump_sum_payment=None,
                     property_value=None, construction_type=None, occupancy_type=None,
                     Secured_by=None, total_units=None, income=None, age=None, Region=None,
                     Security_Type=None):

    # Create the input dataframe
    X_user = create_df_from_inputs(loan_limit=loan_limit, Gender=Gender, open_credit=open_credit,
                     business_or_commercial=business_or_commercial, loan_amount=loan_amount,
                     term=term, interest_only=interest_only, lump_sum_payment=lump_sum_payment,
                     property_value=property_value, construction_type=construction_type, occupancy_type=occupancy_type,
                     Secured_by=Secured_by, total_units=total_units, income=income, age=age, Region=Region,
                     Security_Type=Security_Type)


    # creating_full_dataframe_from_inputs(loan_limit, income, age)

    # Load the preprocessor and transform the input dataframe
    preprocessor = process_data()
    X_user_processed = preprocessor.transform(X_user).drop(columns="Status")

    # Load the model from the pretrain model pickle file
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
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
    my_prediction_function(loan_limit="cf", income=5760.0, age="45-54")




# # OLD

# def creating_full_dataframe_from_inputs(loan_limit, income, age):
#     # Get the path to the Loan_Default.csv file (raw data))
#     ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
#     raw_data_path = os.path.join(ROOT_PATH, 'raw_data', 'Loan_Default.csv')

#     # Printing raw_data_path
#     print('Raw_data_path: ', raw_data_path)

#     # Convert the Loan_Default.file into a DataFrame
#     data_raw = pd.read_csv(raw_data_path)

#     # Creating the input Dataframe
#     X_user = pd.DataFrame(data_raw.iloc[0, :]).transpose()#.drop(columns="Status")
#     X_user["loan_limit"] = loan_limit
#     X_user["income"] = income
#     X_user["age"] = age

#     print("✅ Input dataframe created successfully")

#     return X_user

# def my_prediction_function(loan_limit, income, age):
#     # Create the input dataframe
#     X_user = creating_full_dataframe_from_inputs(loan_limit, income, age)

#     # Load the preprocessor and transform the input dataframe
#     preprocessor = process_data()
#     X_user_processed = preprocessor.transform(X_user).drop(columns="Status")

#     # Load the model from the pretrain model pickle file
#     ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
#     model_path = os.path.join(ROOT_PATH, 'models', 'mvp_model.pkl')
#     print(f"Path of the model.pkl:\n{model_path}\n")
#     with open(model_path, 'rb') as file:
#         model = pickle.load(file)

#     # Predict
#     prediction = model.predict(X_user_processed)
#     print(f"Prediction: {prediction[0]}")

#     print("✅ Prediction done succesfully")

#     return prediction
