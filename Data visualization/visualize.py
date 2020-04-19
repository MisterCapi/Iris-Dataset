import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

sns.set(style="ticks")
sns.PairGrid(iris, hue="species").map_diag(plt.hist).map_offdiag(plt.scatter).add_legend()
plt.show()

print(iris)
