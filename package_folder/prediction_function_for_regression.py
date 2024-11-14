import os
import pathlib
import pickle

# Analysis
import pandas as pd

# Machine learning
from sklearn.linear_model import LogisticRegression

# Import preprocess_light functions
from package_folder.preprocessor_for_regression import *
from package_folder.create_df_from_inputs import *

def prediction_function_for_regression(loan_limit=None, Gender=None, open_credit=None,
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
if __name__ == "__main__": #Test on data.iloc[100] from preprocessor_light
    prediction_function_for_regression(
        loan_limit='cf',
        Gender='Male',
        open_credit='nopc',
        business_or_commercial='nob/c',
        loan_amount=186500,
        term=360,
        interest_only='not_int',
        property_value=348000,
        income=2880,
        age='55-64')

#Comment: predicted interest rate = 3.90% while real value = 4.875%
