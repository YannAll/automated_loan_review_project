import os
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# Load loan data
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

# Clean data
def clean_data(data):
    # Drop duplicate rows
    data = data.drop_duplicates()

    # Remove columns with more than 25% missing values
    missing_percentage = data.isnull().sum() / len(data) * 100
    data = data.loc[:, missing_percentage <= 25]

    # Remove samples with more than 5 missing values
    data['missing_count'] = data.isnull().sum(axis=1)
    data = data[data['missing_count'] <= 5]
    data = data.drop(columns=['missing_count'])

    print("Data cleaned successfully.")
    return data

# Encode categorical variables
def encode_categorical(data):
    # Encode categorical features using OneHotEncoder for categorical variables
    cat_cols = data.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # Fit and transform categorical columns
    encoded_data = pd.DataFrame(encoder.fit_transform(data[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))

    # Drop original categorical columns and concatenate the encoded data
    data = data.drop(cat_cols, axis=1).reset_index(drop=True)
    data = pd.concat([data, encoded_data], axis=1)

    print("Categorical variables encoded successfully.")
    return data

# Impute missing values using KNN
def knn_impute(data):
    # Select numerical columns for KNN Imputation
    num_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]

    # Initialize KNN Imputer
    knn_imputer = KNNImputer(n_neighbors=3)

    # Fit and transform the data
    data[num_cols] = knn_imputer.fit_transform(data[num_cols])

    print("KNN Imputation completed successfully.")
    return data

# Tree-based imputation
def tree_imputation(data):
    # Define columns with missing values
    missing_cols = [col for col in data.columns if data[col].isnull().sum() > 0]
    non_missing_cols = [col for col in data.columns if data[col].isnull().sum() == 0]

    for col in missing_cols:
        # Define a bagging model for each attribute
        model = BaggingRegressor(DecisionTreeRegressor(), n_estimators=40, max_samples=1.0, max_features=1.0, bootstrap=False, n_jobs=-1)

        # Separate rows with and without missing values in the target column
        col_missing = data[data[col].isnull()]
        temp = data.drop(data[data[col].isnull()].index, axis=0)

        # Define features and target
        X = temp[non_missing_cols]
        y = temp[col]

        # Fit the model and predict missing values
        model.fit(X, y)
        y_pred = model.predict(col_missing[non_missing_cols])

        # Impute the missing values
        data.loc[col_missing.index, col] = y_pred

    print("Tree-based imputation completed successfully.")
    return data

# Create preprocessor
def create_preprocessor():
    # Define categorical and numerical columns
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Define transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
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
data = encode_categorical(data)
data = knn_impute(data)
data = tree_imputation(data)
preprocessor = create_preprocessor()
