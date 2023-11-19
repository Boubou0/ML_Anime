import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression

#Evaluer la performance du modèle
from sklearn.metrics import  mean_squared_error, r2_score

df = pd.read_csv('test.csv', delimiter=';', encoding = "ISO-8859-1")



#shuffle df


ohe = OneHotEncoder(sparse_output=False)
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])
df['Source'] = le.fit_transform(df['Source'])

df['Genres'] = ohe.fit_transform(df['Genres'].str.get_dummies(sep=',')).astype(int)
df['Studios'] = ohe.fit_transform(df['Studios'].str.get_dummies(sep=',')).astype(int)

#df = df.sample(frac=1)
matrix = df.to_numpy()

#print 5 lignes de la matrice
#print(matrix[0:5,:])
Y = matrix[:,0]

selector = SelectKBest(f_regression, k=2)
X_new = selector.fit_transform(matrix, Y)

#Séparer les données en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.2, random_state=42)

# Instancier une regression lineaire
reg = LinearRegression()

# Entraîner le modèle
reg.fit(X_train, y_train)

# Prédire les scores
y_pred = reg.predict(X_test)
y_pred_train = reg.predict(X_train)

# Calculer les métriques de performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error : ", mse)
print("R-Squared : ", r2)

# Effectuer la cross-validation en utilisant la métrique R²
# scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='r2')

# Afficher les scores pour chaque itération de la cross-validation
#print("Scores de cross-validation: ", scores)

# Afficher la moyenne des scores
#print("Moyenne des scores: ", np.mean(scores))


# Donc pas de relations linéaires entre les variables et le score