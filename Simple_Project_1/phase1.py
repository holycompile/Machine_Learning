import numpy as np 

def validate_data(x,y):
    if len(x)==len(y):
        return True
    else:
        return False
    
xstudy=np.array([1,2,3,4,5,6])
ymarks=np.array([35, 40, 50, 55, 65, 70])
result=validate_data(xstudy,ymarks)
print("Study hours:",xstudy)
print("Marks:",ymarks)
print("Same length check:", result)
