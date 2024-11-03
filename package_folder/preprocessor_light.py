import os
import pathlib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load loan data
def load_loan_data():
    ROOT_PATH = pathlib.Path().resolve().parent
    raw_data_path = os.path.join(ROOT_PATH, 'raw_data', 'Loan_Default.csv')
    print(f"🔍 Checking for file at path: {raw_data_path}")

    if os.path.exists(raw_data_path):
        data = pd.read_csv(raw_data_path)
        print("✅ Data loaded successfully")
        return data
    else:
        raise FileNotFoundError(f"The file {raw_data_path} does not exist. Please check the path.")

# Step 1: Clean data transformer
class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop duplicate rows
        X = X.drop_duplicates()

        # Remove columns with more than 25% missing values
        missing_percentage = X.isnull().sum() / len(X) * 100
        X = X.loc[:, missing_percentage <= 25]

        print("✅ Data cleaned")
        return X

# Step 2: Drop unnecessary columns
class ColumnDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.drop_columns = ['year', 'ID']
        return self

    def transform(self, X):
        X = X.drop(columns=self.drop_columns, errors='ignore')
        print("✅ Columns 'year' and 'ID' dropped")
        return X

# Step 3: Impute missing values in categorical variables
class CategoricalImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cat_cols = [col for col in X.columns if X[col].dtype == 'object']
        self.imputer = SimpleImputer(strategy='most_frequent')
        self.imputer.fit(X[self.cat_cols])
        return self

    def transform(self, X):
        X[self.cat_cols] = self.imputer.transform(X[self.cat_cols])
        print("✅ Missing values in categorical variables imputed")
        return X

# Step 4: Encode categorical variables
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Cast 'term' to object so it is treated as a categorical variable
        if 'term' in X.columns:
            X['term'] = X['term'].astype(str)

        # Identify all categorical columns in the DataFrame, including 'term'
        self.cat_cols = [col for col in X.columns if X[col].dtype == 'object']

        # Initialize the OneHotEncoder
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.encoder.fit(X[self.cat_cols])
        return self

    def transform(self, X):
        # Cast 'term' to object so it is treated as a categorical variable
        if 'term' in X.columns:
            X['term'] = X['term'].astype(str)

        # Perform One-Hot Encoding for all categorical columns
        encoded_data = pd.DataFrame(self.encoder.transform(X[self.cat_cols]), columns=self.encoder.get_feature_names_out(self.cat_cols))

        # Drop the original categorical columns and concatenate the encoded columns
        X = X.drop(self.cat_cols, axis=1).reset_index(drop=True)
        X = pd.concat([X, encoded_data.reset_index(drop=True)], axis=1)

        print("✅ Categorical variables encoded successfully, including 'term'")
        return X

# Step 5: Impute missing values with Simple Imputer
class SimpleImputerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        self.imputer = SimpleImputer(strategy='mean')
        self.imputer.fit(X[self.num_cols])
        return self

    def transform(self, X):
        # Ensure no NaN values remain after imputation
        X[self.num_cols] = self.imputer.transform(X[self.num_cols])
        X[self.num_cols] = X[self.num_cols].fillna(0)  # Fill remaining NaNs with 0 to prevent issues
        print("✅ Missing values imputed with Simple Imputer (mean), remaining NaNs filled with 0")
        return X

# Step 6: Outlier removal transformer
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, iqr_factor=3):
        self.iqr_factor = iqr_factor

    def fit(self, X, y=None):
        # Identify non-binary numerical columns
        self.numerical_columns = [col for col in X.select_dtypes(include=['float64', 'int64']).columns if X[col].nunique() > 2]
        return self

    def transform(self, X):
        if len(X) < 50:
            print("⚠️ Small dataset detected, skipping outlier removal to avoid excessive data loss")
            return X

        for col in self.numerical_columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_factor * IQR
            upper_bound = Q3 + self.iqr_factor * IQR
            X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
        print("✅ Outliers removed based on IQR threshold")
        return X

# Step 7: Scaling continuous variables
class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.continuous_columns = [col for col in X.select_dtypes(include=['float64', 'int64']).columns if X[col].nunique() > 2 and col != 'term']
        self.scaler = MinMaxScaler()
        self.scaler.fit(X[self.continuous_columns])
        return self

    def transform(self, X):
        # Scale continuous columns
        if self.continuous_columns:
            X[self.continuous_columns] = self.scaler.transform(X[self.continuous_columns])
            print("✅ Continuous variables scaled between 0 and 1")
        else:
            print("⚠️ No continuous variables found to scale")
        return X

# Full preprocessing pipeline
def create_preprocessing_pipeline():
    pipeline = Pipeline([
        ('cleaner', DataCleaner()),
        ('dropper', ColumnDropper()),
        ('cat_imputer', CategoricalImputer()),
        ('encoder', CategoricalEncoder()),
        ('simple_imputer', SimpleImputerTransformer()),
        ('outlier_remover', OutlierRemover()),
        ('scaler', MinMaxScalerTransformer())
    ])
    return pipeline

# Main processing function
def process_data():
    data = load_loan_data()

    # Full pipeline with all steps
    full_pipeline = create_preprocessing_pipeline()

    # Process data through the pipeline
    data_processed = full_pipeline.fit_transform(data)

    # Save the processed data
    output_path = os.path.join(pathlib.Path().resolve().parent, 'raw_data', 'loan_preprocessed.csv')
    data_processed.to_csv(output_path, index=False)
    print(f"✅ Transformed data saved successfully at {output_path}")

# Run the processing pipeline
if __name__ == "__main__":
    process_data()
