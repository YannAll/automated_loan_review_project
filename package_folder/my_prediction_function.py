def my_prediction_function(age,income,loan_limit):
    if int(age)>20 and int(income)>20000 and int(loan_limit)<int(income*10):
        return f"your application for a loan of {loan_limit} EUR is approved"
    else:
        return f"your application for a loan of {loan_limit} EUR is not approved"

print(my_prediction_function(21,30000,500000))
