from fastapi import FastAPI
from package_folder.my_prediction_function import my_prediction_function
import pickle

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {'greeting':"hello"}

# Prediction endpoint
@app.get("/predict")
def predict(age,income,loan_limit):
    # Use the function in our package to run the prediction
    prediction = my_prediction_function(age,income,loan_limit)

    # Return prediction
    return {"prediction": prediction}
