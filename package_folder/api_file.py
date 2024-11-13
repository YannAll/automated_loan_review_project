from fastapi import FastAPI
from package_folder.my_prediction_function import my_prediction_function
from package_folder.prediction_function_for_regression import *
import pickle

# FastAPI instance
app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {'greeting':"hello"}

# Prediction endpoint
@app.get("/predict")
def predict(loan_limit=None,
            Gender=None,
            open_credit=None,
            business_or_commercial=None,
            loan_amount=None,
            term=None,
            interest_only=None,
            lump_sum_payment=None,
            property_value=None,
            construction_type=None,
            occupancy_type=None,
            Secured_by=None,
            total_units=None,
            income=None,
            age=None,
            Region=None,
            Security_Type=None):

    # Use a function in our package to run the application status (accepted=0, declined=1)
    status = int(my_prediction_function(loan_limit=loan_limit,
                                        Gender=Gender,
                                        open_credit=open_credit,
                                        business_or_commercial=business_or_commercial,
                                        loan_amount=loan_amount,
                                        term=term,
                                        interest_only=interest_only,
                                        lump_sum_payment=lump_sum_payment,
                                        property_value=property_value,
                                        construction_type=construction_type,
                                        occupancy_type=occupancy_type,
                                        Secured_by=Secured_by,
                                        total_units=total_units,
                                        income=income,
                                        age=age,
                                        Region=Region,
                                        Security_Type=Security_Type))

    # Use another function in our package to confirm the interest rate to the applicant
    if status==0:
        status='Your credit application is approved'
        interest_rate = round(float(prediction_function_for_regression(loan_limit=loan_limit,
                                        Gender=Gender,
                                        open_credit=open_credit,
                                        business_or_commercial=business_or_commercial,
                                        loan_amount=loan_amount,
                                        term=term,
                                        interest_only=interest_only,
                                        lump_sum_payment=lump_sum_payment,
                                        property_value=property_value,
                                        construction_type=construction_type,
                                        occupancy_type=occupancy_type,
                                        Secured_by=Secured_by,
                                        total_units=total_units,
                                        income=income,
                                        age=age,
                                        Region=Region,
                                        Security_Type=Security_Type)),2)

    else:
        status='Your credit application is not approved'
        interest_rate='Not applicable'

    result = {"status": status, "interest_rate": f"{interest_rate}%"}

    # Return status and interest rate to user
    return result


# # # OLD

# # Root endpoint
# @app.get("/")
# def root():
#     return {'greeting':"hello"}

# # Prediction endpoint
# @app.get("/predict")
# def predict(age,income,loan_limit):
#     # Use the function in our package to run the prediction
#     prediction = my_prediction_function(age,income,loan_limit)

#     # Return prediction
#     return {"prediction": prediction}
