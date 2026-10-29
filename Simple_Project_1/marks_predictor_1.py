import numpy as np 

def validate_data(x,y):
    if len(x)==len(y):
        return True
    else:
        return False

def predict_marks(m):
    return (5*m)+30
    
xstudy=np.array([1,2,3,4,5,6])

ymarks=np.array([35, 40, 50, 55, 65, 70])

error=[]

iteration=101
k=0
for i in xstudy:
    print("Iteration No:",iteration)
    print("Study Hours: ",i)
    predicted_marks_=predict_marks(i)
    print("Predicted Marks:",predicted_marks_)
    print("Actual marks",ymarks[k])
    error_=abs(ymarks[k]-predicted_marks_)
    print("Error: ",error_)
    error.append(error_)
    k=k+1
    iteration+=1
    print("--------------------------------------------")
    
print("Net error: ",error)
# Mean error
print("Mean error:",np.mean(error) )