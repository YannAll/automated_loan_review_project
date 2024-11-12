# General
import os
import pathlib
import pickle

# Analysis
import numpy as np
import pandas as pd

# Machine learning
from sklearn.pipeline import Pipeline


def create_df_from_inputs(loan_limit=None, Gender=None, open_credit=None,
                     business_or_commercial=None, loan_amount=None,
                     term=None, interest_only=None, lump_sum_payment=None,
                     property_value=None, construction_type=None, occupancy_type=None,
                     Secured_by=None, total_units=None, income=None, age=None, Region=None,
                     Security_Type=None):

    # Load the model from the fitted preprocessor pickle file
    # ROOT_PATH = pathlib.Path().resolve().parent # Get the parent directory of the current working directory
    # preprocessor_input_path = os.path.join(ROOT_PATH, 'models', 'preprocessor_fitted_input.pkl')

    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    preprocessor_input_path = os.path.join(ROOT_PATH, 'models', 'preprocessor_fitted_input.pkl')

    # print(f"Path of the preprocessor.pkl:\n{preprocessor_input_path}\n")
    with open(preprocessor_input_path, 'rb') as file:
        preprocessor = pickle.load(file)

    columns_names = [column_name[17:] for column_name in preprocessor.get_feature_names_out()]

    # Creating a NaN dataframe ready to be imputed
    df_nan = pd.DataFrame(np.nan, index = np.arange(1), columns =columns_names)

    # Impute the Nan with the preprocessor
    df_base = pd.DataFrame(preprocessor.transform(df_nan), index=np.arange(1), columns=preprocessor.get_feature_names_out())

    # Remove the first 17 characters from each column name
    df_base.columns = df_base.columns.str.slice(17)

    # Creating a dict from the inputs
    dict = {
        'loan_limit': loan_limit,
        'Gender': Gender,
        'open_credit': open_credit,
        'business_or_commercial': business_or_commercial,
        'loan_amount': loan_amount,
        'term': term,
        'interest_only': interest_only,
        'lump_sum_payment': lump_sum_payment,
        'property_value': property_value,
        'construction_type': construction_type,
        'occupancy_type': occupancy_type,
        'Secured_by': Secured_by,
        'total_units': total_units,
        'income': income,
        'age': age,
        'Region': Region,
        'Security_Type': Security_Type
    }

    # Creating a Dataframe from the dict
    df_input = pd.DataFrame(dict, index = np.arange(1))

    # Get columns names

    column_names = list(df_input.columns)

    # Replace df_base values by inputs values if not None
    for column_name in column_names:
        if df_input.at[0, column_name] is not None:
            df_base.at[0, column_name] = df_input.loc[0, column_name]

    print("✅ Input dataframe succesfully created")

    return df_base

# Run the processing pipeline
if __name__ == "__main__":
    create_df_from_inputs(loan_limit="cf", income=5760.0, age="45-54")
