import numpy as np 

def validate_data(x,y):
    if len(x)==len(y):
        return True
    else:
        return False

def predict_marks_model_A(study_hrs, slope, inter):
    return (slope*study_hrs)+inter

def predict_marks_model_B(study_hrs, slope, inter):
    return (slope*study_hrs)+inter
    
xstudy=np.array([1,2,3,4,5,6])

ymarks=np.array([35, 40, 50, 55, 65, 70])



models= {
    "Model A":
        {
            "name":"A",
            "slope": 5,
            "intercept": 30,
            "error": []
        },
    "Model B":
        {
           "name":"B",
            "slope": 6,
            "intercept": 30,
            "error":[]
        }
}

mean_error_of_models=[]



for model_info in models:
    print("Current Model : ",model_info)
    print("Slope :", models[model_info]["slope"])
    print("Intercept :", models[model_info]["intercept"])
    k=0
    for study_hours in xstudy:
        actual_marks=ymarks[k]
        if models[model_info]["name"] == "A":
            predicted_marks= predict_marks_model_A(study_hours,models[model_info]["slope"], models[model_info]["intercept"])
        else:
            predicted_marks= predict_marks_model_B(study_hours,models[model_info]["slope"], models[model_info]["intercept"])
        
        #Calculating the error of that model's iteration 
        models[model_info]["error"].append(abs(actual_marks-predicted_marks))
        print(f"Study hours {study_hours} | Actual Marks: {actual_marks} | Predicted Marks: {predicted_marks} | Error: {models[model_info]["error"][k]}")
        
        k+=1
    mean_error=np.mean(models[model_info]["error"])
    print("Mean error of this model : ",mean_error)
    mean_error_of_models.append(mean_error)
    print("-----------------------------------------------------------------------------------------------------------------")
    
    