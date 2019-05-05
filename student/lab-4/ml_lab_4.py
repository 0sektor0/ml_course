# -*- coding: utf-8 -*-
"""ml-lab-4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12LCJOjXZ49GbErpCRSCNWHNWymqLFUo8

# **Задание:**


1.   Выберите набор данных (датасет) для решения задачи классификации или регресии.
2.   В случае необходимости проведите удаление или заполнение пропусков и кодирование категориальных признаков.
1.   С использованием метода train_test_split разделите выборку на обучающую и тестовую.
2.   Обучите модель ближайших соседей для произвольно заданного гиперпараметра K. Оцените качество модели с помощью трех подходящих для задачи метрик.
1.   Постройте модель и оцените качество модели с использованием кросс-валидации. Проведите эксперименты с тремя различными стратегиями кросс-валидации.
2.   Произведите подбор гиперпараметра K с использованием GridSearchCV и кросс-валидации.
1.   Повторите пункт 4 для найденного оптимального значения гиперпараметра K. Сравните качество полученной модели с качеством модели, полученной в пункте 4.
2.   Постройте кривые обучения и валидации.

Датасет: [wine](https://www.kaggle.com/brynja/wineuci/downloads/wineuci.zip/1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut, LeavePOut, ShuffleSplit, StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import roc_curve,confusion_matrix, roc_auc_score, accuracy_score, balanced_accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# Считывание данных
data = pd.read_csv('Wine.csv', sep=";")
data.head()

# Типы данных
data.dtypes

# Проверка на пустые значения
for col in data.columns:
    print('{} - {}'.format(col, data[data[col].isnull()].shape[0]))

# Размерность данных
data.shape

"""# **Разделим выборку при помощи train_test_split**"""

CLASS = 'Class'
RANDOM_STATE = 17
TEST_SIZE = 0.3

X = data.drop(CLASS, axis=1).values
Y = data[CLASS].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y)
print('X_train: {}'.format(X_train.shape))
print('X_test: {}'.format(X_test.shape))

"""# **Обучение ан различном числе соседей и оценка качества**"""

# Задаем число соседей
NEIGHBOURS_MAX_COUNT = 50
neighbours_count = np.arange(1, NEIGHBOURS_MAX_COUNT+1)

train_accuracy =np.empty(NEIGHBOURS_MAX_COUNT)
test_accuracy = np.empty(NEIGHBOURS_MAX_COUNT)

for i, k in enumerate(neighbours_count):
    # Настройка классификатора Knn с K соседями
    knn = KNeighborsClassifier(n_neighbors = k)
    
    # Обучить модель
    knn.fit(X_train, Y_train)
    
    # Вычислить точность на тренировочном наборе
    train_accuracy[i] = knn.score(X_train, Y_train)
    
    # Вычислить точность на тестовом наборе
    test_accuracy[i] = knn.score(X_test, Y_test)
    
# Построить набор
plt.title('k-NN различное число соседей')
plt.plot(neighbours_count, test_accuracy, label='Тестовая точность')
plt.plot(neighbours_count, train_accuracy, label='Обучающая точность')
plt.legend()
plt.xlabel('Число соседей')
plt.ylabel('Точность')
plt.show()

# Обучение и оценка качества
knn = KNeighborsClassifier(n_neighbors = 20)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
print(classification_report(Y_test, Y_pred))

"""# **Постройте модель и оцените качество модели с использованием кросс-валидации**"""

CROSS_VALIDATOR_GENERATOR = 5
N_NEIGHBOURS_TAG = 'n_neighbors' 

param_grid = {N_NEIGHBOURS_TAG : np.arange(1, NEIGHBOURS_MAX_COUNT + 1)}
knn = KNeighborsClassifier()

knn_cv= GridSearchCV(knn, param_grid, cv = CROSS_VALIDATOR_GENERATOR)
knn_cv.fit(X_train,Y_train)


knn_cv.best_score_

Y_pred = knn_cv.predict(X_test)
print(classification_report(Y_test, Y_pred))

knn_cv.best_params_

"""K-fold
Данная стратегия работает в соответствии с определением кросс-валидации.

Каждой стратегии в scikit-learn ставится в соответствии специальный класс-итератор, который может быть указан в качестве параметра cv функций cross_val_score и cross_validate.
"""

FOLDS_COUNT = 5
BEST_PARAMS = knn_cv.best_params_[N_NEIGHBOURS_TAG]

knn = KNeighborsClassifier(n_neighbors = BEST_PARAMS)
cv = KFold(n_splits = FOLDS_COUNT)
scores = cross_val_score(knn, X, Y, cv = cv)

np.mean(scores)

"""Leave One Out (LOO)
В тестовую выборку помещается единственный элемент (One Out). Количество фолдов в этом случае определяется автоматически и равняется количеству элементов.

Данный метод более ресурсоемкий чем KFold.

Существует эмпирическое правило, что вместо Leave One Out лучше использовать KFold на 5 или 10 фолдов.
"""

loo = LeaveOneOut()
loo.get_n_splits(X)

for train_index, test_index in loo.split(X):
   Y_train, Y_test = Y[train_index], Y[test_index]
    
knn = KNeighborsClassifier(n_neighbors = BEST_PARAMS)
scores = cross_val_score(knn, X, Y, cv = loo)

np.mean(scores)

"""Repeated K-Fold"""

knn = KNeighborsClassifier(n_neighbors = BEST_PARAMS)
cv = RepeatedKFold(n_splits = FOLDS_COUNT, n_repeats = 2)
scores = cross_val_score(knn, X, Y, cv = cv)

np.mean(scores)

"""# **Постройте кривые обучения и валидации.**"""

# Кривые обучения
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
  
  
knn = KNeighborsClassifier(n_neighbors = 4)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=Y)

plot_learning_curve(knn, 'n_neighbors=4', X_train, Y_train, cv=5)

# Кривая валидации
def plot_validation_curve(estimator, title, X, y, 
                          param_name, param_range, cv, 
                          scoring="accuracy"):
                                                   
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    return plt

n_range = np.array(range(5,55,5))
plot_validation_curve(KNeighborsClassifier(n_neighbors=4), 'knn', 
                      X_train, y_train, 
                      param_name='n_neighbors', param_range=n_range, 
                      cv=5, scoring="accuracy")