#!pip install seaborn --upgrade

from google.colab import files
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline 
sns.set(style="ticks")

# load file to analyze
#uploaded = files.upload()
  
# Будем анализировать данные только на обучающей выборке
data = pd.read_csv('mall-customers.csv', sep=",")

#параметры модели
customerId = 'CustomerID'           # Unique ID assigned to the customer
gender = 'Gender'                   # Gender of the customer
age = 'Age'                         # Age of the customer
annualIncome = 'Annual Income'      # Annual Income of the customee
customerScore = 'Spending Score'    # Score assigned by the mall based on customer behavior and spending nature

# В качестве целевого признака выберем Annual Income 

# Первые 5 строк датасета
data.head()

# Размер датасета 
data.shape

total_count = data.shape[0]
print('data at all: {}'.format(total_count))

# Список колонок
data.columns

# Список колонок с типами данных
data.dtypes

# Проверим наличие пустых значений
# Цикл по колонкам датасета
for col in data.columns:
    # Количество пустых значений - все значения заполнены
    temp_null_count = data[data[col].isnull()].shape[0]
    print('{} - {}'.format(col, temp_null_count))

# Основные статистические характеристки набора данных
data.describe()

# Определим уникальные значения для целевого признака
data[gender].unique()

# Диаграмма рассеяния
figure, axes = plt.subplots(figsize=(10,10)) 
sns.scatterplot(axes=axes, x=customerScore, y=annualIncome, data=data, hue=customerScore)

# Joint plot
sns.jointplot(x=age, y=customerScore, data=data, kind="kde", space=0, color="r")

# Joint plot
sns.jointplot(x=age, y=annualIncome, data=data, kind="kde", space=0, color="r")

# Парные диаграммы
sns.pairplot(data, hue=age)

# Распределение параметра Gender сгруппированные по SpendingScore.
sns.boxplot(x=gender, y=annualIncome, data=data)

# Violin plot
figure, axes = plt.subplots(2, 1, figsize=(10,10))
sns.violinplot(ax=axes[0], x=data[annualIncome])
sns.distplot(data[annualIncome], ax=axes[1])

# Heat map
# Треугольный вариант матрицы
mask = np.zeros_like(data.corr(), dtype=np.bool)
# чтобы оставить верхнюю часть матрицы
mask[np.tril_indices_from(mask)] = True
sns.heatmap(data.corr(), mask=mask, annot=True, fmt='.3f')

"""Из результатов анализа видно, данные слабо связанны друг с другом"""