import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from sklearn import metrics

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

matrix = [[471623,206],[26223,240]]

heatmap = sns.heatmap(matrix,
                       annot=True,
                       fmt="d",
                       cbar=False,
                       cmap="spring")

plt.title('Ridge Regression with ')
plt.ylabel('Real')
plt.xlabel('Predicted')
plt.show()