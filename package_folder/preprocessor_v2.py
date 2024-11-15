import os
import pathlib
import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Define the main project directory
ROOT_PATH = pathlib.Path(__file__).resolve().parents[1]
log_file_path = ROOT_PATH / 'preprocessing.log'

# Set up logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load loan data
def load_loan_data():
    raw_data_path = ROOT_PATH / 'raw_data' / 'Loan_Default.csv'
    logging.info(f"Checking for file at path: {raw_data_path}")

    try:
        data = pd.read_csv(raw_data_path)
        logging.info(f"Data loaded successfully with shape {data.shape}")
        return data
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        raise

# Step 1: Remove duplicates
class DataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            initial_row_count = X.shape[0]
            X = X.drop_duplicates()
            removed_rows = initial_row_count - X.shape[0]
            logging.info(f"Removed {removed_rows} duplicate rows. Data shape: {X.shape}")
            return X
        except Exception as e:
            logging.error(f"Error in DataCleaner: {e}")
            raise

# Step 2: Convert `term` to categorical type
class TermCategorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            if 'term' in X.columns:
                X['term'] = X['term'].astype(str)
            logging.info("Converted `term` to categorical type")
            return X
        except Exception as e:
            logging.error(f"Error in TermCategorizer: {e}")
            raise

# Step 3: Drop specific columns
class ColumnDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.drop_columns = ['year', 'ID', 'Interest_rate_spread', 'Upfront_charges']
        return self

    def transform(self, X):
        try:
            X = X.drop(columns=self.drop_columns, errors='ignore')
            logging.info(f"Dropped columns: {self.drop_columns}. Data shape: {X.shape}")
            return X
        except Exception as e:
            logging.error(f"Error in ColumnDropper: {e}")
            raise

# Step 4: Calculate the `interest_income_ratio`
class InterestIncomeRatioCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X['interest_income_ratio'] = np.where(
                X['income'] > 0,
                (X['rate_of_interest'] * X['loan_amount']) / X['income'],
                -1  # Placeholder for missing income
            )
            logging.info(f"Calculated `interest_income_ratio`. Data shape: {X.shape}")
            return X
        except Exception as e:
            logging.error(f"Error in InterestIncomeRatioCalculator: {e}")
            raise

# Step 5: Create missing value dummies
class MissingDummyCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X['LTV_missing'] = X['LTV'].isnull().astype(int)
            X['dtir1_missing'] = X['dtir1'].isnull().astype(int)
            X['income_missing'] = X['income'].isnull().astype(int)
            logging.info("Created missing value dummies for LTV, dtir1, and income. Data shape: {X.shape}")
            return X
        except Exception as e:
            logging.error(f"Error in MissingDummyCreator: {e}")
            raise

# Step 6: Set placeholders for missing values
class PlaceholderSetter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X['LTV'] = X['LTV'].fillna(-1)
            X['dtir1'] = X['dtir1'].fillna(-1)
            X['income'] = X['income'].fillna(-1)
            logging.info("Set placeholders for missing values. Data shape: {X.shape}")
            return X
        except Exception as e:
            logging.error(f"Error in PlaceholderSetter: {e}")
            raise

# Step 7: Apply KNN Imputation for other numerical variables
class KNNImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        self.num_cols = [col for col in X.select_dtypes(include=['float64', 'int64']).columns if col not in ['LTV', 'dtir1', 'income']]
        self.knn_imputer.fit(X[self.num_cols])
        return self

    def transform(self, X):
        try:
            X[self.num_cols] = self.knn_imputer.transform(X[self.num_cols])
            logging.info(f"Applied KNN imputation for numerical columns. Data shape: {X.shape}")
            return X
        except Exception as e:
            logging.error(f"Error in KNNImputerTransformer: {e}")
            raise

# Step 8: Outlier removal with IQR
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, iqr_factor=3):
        self.iqr_factor = iqr_factor

    def fit(self, X, y=None):
        self.numerical_columns = [col for col in X.select_dtypes(include=['float64', 'int64']).columns if X[col].nunique() > 2]
        return self

    def transform(self, X):
        try:
            initial_row_count = X.shape[0]
            for col in self.numerical_columns:
                if col == 'Status':
                    continue  # Exclude target variable
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.iqr_factor * IQR
                upper_bound = Q3 + self.iqr_factor * IQR
                X = X[(X[col] >= lower_bound) & (X[col] <= upper_bound)]
            removed_rows = initial_row_count - X.shape[0]
            logging.info(f"Outliers removed based on IQR threshold, {removed_rows} rows dropped. Data shape: {X.shape}")
            return X
        except Exception as e:
            logging.error(f"Error in OutlierRemover: {e}")
            raise
# Step 9: Impute missing values in categorical variables
class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='most_frequent'):
        self.strategy = strategy
        self.imputer = None

    def fit(self, X, y=None):
        # Find all categorical columns
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        if self.strategy == 'most_frequent':
            self.imputer = SimpleImputer(strategy='most_frequent')
            self.imputer.fit(X[self.cat_cols])
        return self

    def transform(self, X):
        try:
            if self.strategy == 'most_frequent':
                X[self.cat_cols] = self.imputer.transform(X[self.cat_cols])
            elif self.strategy == 'placeholder':
                X[self.cat_cols] = X[self.cat_cols].fillna('missing')
            logging.info(f"Imputed missing values in categorical variables using strategy: {self.strategy}")
            return X
        except Exception as e:
            logging.error(f"Error in CategoricalImputer: {e}")
            raise

# Step 10: Categorical encoding
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.encoder.fit(X[self.cat_cols])
        return self

    def transform(self, X):
        try:
            encoded_data = pd.DataFrame(self.encoder.transform(X[self.cat_cols]), columns=self.encoder.get_feature_names_out(self.cat_cols))
            X = X.drop(self.cat_cols, axis=1).reset_index(drop=True)
            X = pd.concat([X, encoded_data.reset_index(drop=True)], axis=1)
            logging.info(f"Categorical variables encoded. Data shape: {X.shape}")
            return X
        except Exception as e:
            logging.error(f"Error in CategoricalEncoder: {e}")
            raise

# Step 11: Scaling
class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.continuous_columns = [col for col in X.select_dtypes(include=['float64', 'int64']).columns if col != 'Status']
        self.scaler = MinMaxScaler()
        self.scaler.fit(X[self.continuous_columns])
        return self

    def transform(self, X):
        try:
            X[self.continuous_columns] = self.scaler.transform(X[self.continuous_columns])
            logging.info(f"Continuous variables scaled. Data shape: {X.shape}")
            return X
        except Exception as e:
            logging.error(f"Error in MinMaxScalerTransformer: {e}")
            raise

# Full preprocessing pipeline
def create_preprocessing_pipeline():
    pipeline = Pipeline([
        ('cleaner', DataCleaner()),
        ('term_categorizer', TermCategorizer()),
        ('dropper', ColumnDropper()),
        ('interest_income_ratio_calculator', InterestIncomeRatioCalculator()),
        ('missing_dummy', MissingDummyCreator()),
        ('placeholder', PlaceholderSetter()),
        ('knn_imputer', KNNImputerTransformer()),
        ('outlier_remover', OutlierRemover()),
        ('categorical_imputer', CategoricalImputer(strategy='placeholder')), # Replace missing categories with 'missing'
        ('encoder', CategoricalEncoder()),
        ('scaler', MinMaxScalerTransformer())
    ])
    return pipeline

# Main processing function
def process_data():
    data = load_loan_data()
    initial_rows, initial_cols = data.shape

    try:
        full_pipeline = create_preprocessing_pipeline()
        data_processed = full_pipeline.fit_transform(data)
        logging.info("Data processed successfully through the pipeline.")
    except Exception as e:
        logging.error(f"Error during pipeline processing: {e}")
        raise

    # Save the processed data
    ROOT_PATH = pathlib.Path(__file__).resolve().parents[1]
    output_path = ROOT_PATH / 'raw_data' / 'loan_preprocessed.csv'
    try:
        data_processed.to_csv(output_path, index=False)
        final_rows, final_cols = data_processed.shape
        row_percentage = (final_rows / initial_rows) * 100
        logging.info(f"Final dataset contains {final_cols} variables and {final_rows} rows ({row_percentage:.2f}% of original rows).")
        logging.info(f"Transformed data saved successfully at {output_path}")
    except Exception as e:
        logging.error(f"Error while saving processed data: {e}")
        raise

    return data_processed

# Run the processing pipeline
if __name__ == "__main__":
    process_data()
