import numpy as np
import matplotlib.pyplot as plt

random_array=np.random.normal(5.0, 1.0, 100000)
print(random_array)
plt.hist(random_array,100)
plt.show()