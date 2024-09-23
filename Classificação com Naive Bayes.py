#CLASSIFICAÇÃO COM NAIVE BAYES
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import seaborn as sns #seaborn serve para visualizar dados

# Carregamento da base de dados

dataset = pd.read_csv('Numpy e Estatística/credit_data.csv')

dataset.dropna(inplace = True) #apagar linhas com dados vazios
# print(dataset.shape)

# sns.countplot(x=dataset['c#default'])
# plt.show()

X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values
# print(X)
# print(y)

# Base de treinamento e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = 1)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)]
print(np.unique(y, return_counts=True)) #verifica quantos valores existem de 0 e 1 na variável y
print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

# print(283/len(y)) verifica a proporção nas bases de dados
# print(226/len(y_train))
# print(57/len(y_test))

# Clssificação com Nayve Bayes

modelo = GaussianNB()
modelo.fit(X_train, y_train)
previsao = modelo.predict(X_test)
# print(previsao)

accuracy = accuracy_score(y_test, previsao) #verifica a precisão do modelo
print(accuracy)

cm = confusion_matrix(y_test, previsao) #indica quantos registros foram classificados corretamente ou não
print(cm)

sns.heatmap(cm, annot=True)
plt.show()
'''
# Subamostragem (undersampling) - Tomek Link
tml = TomekLinks(sampling_strategy = 'majority') # majority = 'apaga registros da classe majoritária'
X_under, y_under = tml.fit_resample(X, y)

# print(X_under.shape, y_under.shape)
# print(np.unique(y, return_counts=True))
# print(np.unique(y_under, return_counts=True))

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_under, y_under, test_size=0.2, 
                                                            stratify=y_under, random_state = 1)
modelo_u = GaussianNB()
modelo_u.fit(X_train_u, y_train_u)
previsao_u = modelo_u.predict(X_test_u)
# print(previsao_u)

accuracy_u = accuracy_score(y_test_u, previsao_u) # verifica a precisão do modelo
print(accuracy_u)

cm_u = confusion_matrix(y_test_u, previsao_u) # indica quantos registros foram classificados corretamente ou não
print(cm_u)

print(313/(313+22)) # porcentagem de acerto com relação aos que pagam -> 0
print(35/(35+10)) # porcentagem de acerto para aqueles que não pagam -> 1
'''

#Sobreamostragem (oversampling) - SMOTE
smote = SMOTE(sampling_strategy='minority')
X_over, y_over = smote.fit_resample(X,y)

# print(X_over.shape, y_over.shape)
# print(np.unique(y, return_counts=True))
# print(np.unique(y_over, return_counts=True))

X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_over, y_over, test_size=0.2, 
                                                            stratify=y_over, random_state = 1)

modelo_o = GaussianNB()
modelo_o.fit(X_train_o, y_train_o)
previsao_o = modelo_o.predict(X_test_o)
# print(previsao_o)

accuracy_o = accuracy_score(y_test_o, previsao_o) # verfica a precisão do modelo
print(accuracy_o)

cm_o = confusion_matrix(y_test_o, previsao_o) # indica quantos registros foram classificados corretamente ou não
print(cm_o)

print(307/(307+21)) # porcentagem de acerto com relação aos que pagam -> 0
print(322/(322+36)) # porcentagem de acerto para aqueles que não pagam -> 1