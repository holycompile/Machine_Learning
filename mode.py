#mode means the repearting value 
from scipy import stats
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
mode=stats.mode(speed)
print("Mode:", mode)