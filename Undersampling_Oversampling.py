#TEORIA

# Oversampling: aumenta o número de exemplos da classe minoritária; método SMOTE
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
'''

# Undersampling: diminui a quantidade de exemplos da classe majoritária; método Tomek Links

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