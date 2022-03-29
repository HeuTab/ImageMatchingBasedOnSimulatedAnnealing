import matplotlib.pyplot as plt
import numpy as np
import math

x = np.arange(0, 0.2, 0.001)

# y = 2 ** x
y = 1/(1+math.e**(-50*(x-0.02)))

# plt.title("指数函数")
plt.plot(x, y)
plt.show()
