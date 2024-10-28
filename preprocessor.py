import os
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_loan_data():
    # Create a booster path
    base_dir = Path.home() / "code" / "YannAll" / "raw_data"

    # Full path to the CSV file
    csv_file_path = base_dir / "Loan_Default.csv"

    # Load the data into a DataFrame
    if csv_file_path.exists():
        data = pd.read_csv(csv_file_path)
        print("Data loaded successfully.")
        return data
    else:
        raise FileNotFoundError(f"The file {csv_file_path} does not exist. Please check the path.")

def clean_data(data):
    # Drop duplicate rows
    data = data.drop_duplicates()

    # Handle missing values for numerical columns
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

    # Handle missing values for categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns
    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])

    print("Data cleaned successfully.")
    return data

def create_preprocessor():
    # Define categorical and numerical columns
    categorical_features = ['loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 'Credit_Worthiness',
                            'open_credit', 'business_or_commercial', 'Neg_ammortization', 'interest_only',
                            'lump_sum_payment', 'construction_type', 'occupancy_type', 'Secured_by', 'total_units',
                            'credit_type', 'co-applicant_credit_type', 'age', 'submission_of_application', 'Region',
                            'Security_Type']
    numerical_features = ['ID', 'year', 'loan_amount', 'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges',
                          'term', 'property_value', 'income', 'Credit_Score', 'LTV', 'Status', 'dtir1']

    # Define transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

# Call
data = load_loan_data()
data = clean_data(data)
preprocessor = create_preprocessor()
