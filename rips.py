from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

data = datasets.make_circles(n_samples=100)[0] + 5 * datasets.make_circles(n_samples=100)[0]
data1 = datasets.make_circles(n_samples=100)[0]

plt.plot(data1)
plt.show()

dgms = ripser(data)['dgms']
plot_diagrams(dgms, show=True)