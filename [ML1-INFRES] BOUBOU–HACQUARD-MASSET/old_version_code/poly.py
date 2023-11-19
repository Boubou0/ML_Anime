import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

#Evaluer la performance du modèle
from sklearn.metrics import r2_score

df = pd.read_csv('anime2.csv', delimiter=';', encoding = "ISO-8859-1")



#shuffle df


ohe = OneHotEncoder(sparse_output=False)
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])
df['Source'] = le.fit_transform(df['Source'])

df['Genres'] = ohe.fit_transform(df['Genres'].str.get_dummies(sep=',')).astype(int)
df['Studios'] = ohe.fit_transform(df['Studios'].str.get_dummies(sep=',')).astype(int)

df = df.sample(frac=1)
matrix = df.to_numpy()

#print 5 lignes de la matrice
#print(matrix[0:5,:])
Y = matrix[:,0]

#Séparer les données en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(matrix[:,1:], Y, test_size=0.2, random_state=42)

# Instancier un modèle de régression polynomiale
reg = PolynomialFeatures(degree=4)
X_poly = reg.fit_transform(X_train)
reg = LinearRegression().fit(X_poly, y_train)

# Effectuer la cross-validation en utilisant la métrique R²
scores = cross_val_score(reg, X_poly, y_train, cv=5, scoring='r2')

# Afficher les scores pour chaque itération de la cross-validation
print("Scores de cross-validation: ", scores)

# Afficher la moyenne des scores
print("Moyenne des scores: ", np.mean(scores))