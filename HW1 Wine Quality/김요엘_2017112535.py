import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
sns.set(color_codes=True)

names = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide",
         "total sulfur dioxide","density","pH","sulphates","alcohol","quality", "WineType"]
red = pd.read_csv(r'/Users/knuprime025/Documents/Python/DataScienceProject/winequality-red.csv',
                  header=13, sep=';', names=names)
white = pd.read_csv(r'/Users/knuprime025/Documents/Python/DataScienceProject/winequality-white.csv',
                  header=13, sep=';', names=names)
red['WineType'] = 'red'
white['WineType'] = 'white'

dataset = red.append(white, ignore_index=True)
dataset.to_csv('winequality.csv', index=False)

dataset = pd.read_csv('./winequality.csv')

for i in range(0, 12):
    print(names[i] + " : ")
    print('mean is {}'.format(dataset[names[i]].mean()))
    print('variance is {}'.format(dataset[names[i]].var()))
    print()

sizes = [len(red), len(white)]
labels = ['red', 'white']
colors = ['red', 'gray']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.lf%%')
plt.title('WineType')
plt.show()

dataset['quality'].plot.hist(subplots=True, alpha=0.5, figsize=(8, 4))
plt.title('Quality')
plt.show()

dataset.boxplot(column=['alcohol'], by='quality')
plt.show()

sns.violinplot(data=dataset, x='WineType', y='fixed acidity')
plt.title('Violin Plot of Wine Dataset')
plt.show()

dataset.plot.scatter(x='residual sugar', y='chlorides')
plt.show()

labels = ['density','pH', 'sulphates']
df = dataset
scatter_matrix(df[labels], alpha=0.5)
plt.show()

