import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 100)

y1 = (700-20*x)/50

y2 = 20 - x

y3 =  (700-50*x)/20

plt.plot(x, y1, label=r'$50x + 20y = 700$')
plt.plot(x, y2, label=r'$x + y = 20$')
plt.plot(x, y3, label=r'$50x + 20y = 700$')

plt.title('Graph of $50x + 20y = 700$ and $x + y = 20$')
plt.grid(True)
plt.legend()
plt.show()
