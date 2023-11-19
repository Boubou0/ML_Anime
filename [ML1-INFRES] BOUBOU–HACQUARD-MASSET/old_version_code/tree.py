import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
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

# Instancier un modèle d'arbre de décision
three = DecisionTreeRegressor()

# Entraîner le modèle
three.fit(X_train, y_train)

# Prédire les scores
y_pred = three.predict(X_test)
y_pred_train = three.predict(X_train)

# # Evaluer la performance du modèle
# print(r2_score(y_test, y_pred))

# print(r2_score(y_train, y_pred_train))

# Effectuer la cross-validation en utilisant la métrique R²
scores = cross_val_score(three, X_train, y_train, cv=5, scoring='r2')

# Afficher les scores pour chaque itération de la cross-validation
print("Scores de cross-validation: ", scores)

# Afficher la moyenne des scores
print("Moyenne des scores: ", np.mean(scores))


plt.axis([10, 20, 10, 20])
plot_tree(three, feature_names= ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'], class_names=["MEDV"],filled=True)
plt.show()


# Donc pas de relations linéaires entre les variables et le score