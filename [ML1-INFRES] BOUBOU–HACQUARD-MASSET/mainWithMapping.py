import numpy as np
import pandas as pd
import json

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('anime_new.csv', delimiter=';')

mappings = {}

mappings['Type'] = pd.factorize(df.loc[:, 'Type'])[1].tolist()
df.loc[:, 'Type'] = pd.factorize(df.loc[:, 'Type'])[0]

mappings['Studios'] = pd.factorize(df.loc[:, 'Studios'])[1].tolist()
df.loc[:, 'Studios'] = pd.factorize(df.loc[:, 'Studios'])[0]

mappings['Source'] = pd.factorize(df.loc[:, 'Source'])[1].tolist()
df.loc[:, 'Source'] = pd.factorize(df.loc[:, 'Source'])[0]

# Convert the dictionary to a JSON object
json_obj = json.dumps(mappings)

# Write the JSON object to a file
with open('mapping2.json', 'w') as f:
    f.write(json_obj)


ohe = OneHotEncoder(sparse_output=False)
le = LabelEncoder()
df['Genres'] = ohe.fit_transform(df['Genres'].str.get_dummies(sep=',')).astype(int)

#genreDf = pd.DataFrame({'Genres': [["Action", "Adventure", "Avant Garde", "Award Winning", "Boys Love", "Comedy", "Drama", "Fantasy", "Girls Love", "Gourmet", "Horror", "Mystery", "Romance", "Sci-Fi", "Slice of Life", "Sports", "Supernatural", "Suspense", "Ecchi", "Erotica", "Hentai", "Adult Cast", "Anthropomorphic", "CGDCT", "Childcare", "Combat Sports", "Crossdressing", "Delinquents", "Detective", "Educational", "Gag Humor", "Gore", "Harem", "High Stakes Game", "Historical", "Idols (Female)", "Idols (Male)", "Isekai", "Iyashikei", "Love Polygon", "Magical Sex Shift", "Mahou Shoujo", "Martial Arts", "Mecha", "Medical", "Military", "Music", "Mythology", "Organized Crime", "Otaku Culture", "Parody", "Performing Arts", "Pets", "Psychological", "Racing", "Reincarnation", "Reverse Harem", "Romantic Subtext", "Samurai", "School", "Showbiz", "Space", "Strategy Game", "Super Power", "Survival", "Team Sports", "Time Travel", "Vampire", "Video Game", "Visual Arts", "Workplace", "Demographics", "Josei", "Kids", "Seinen", "Shoujo", "Shounen"]]})

column = ['Genres', 'Type', 'Episodes', 'Studios', 'Source']
Y = df['Score']
X = df[column]

#Séparer les données en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Instancier un modèle de régression linéaire
reg = LinearRegression()

#Entraîner le modèle
reg.fit(X_train, y_train)

#Prédire les scores
y_pred = reg.predict(X_test)

y_pred_train = reg.predict(X_train)

#Evaluer la performance du modèle
print("\nModèle linéaire")
print("Performance du modèle linéaire en apprentissage : ", r2_score(y_train, y_pred_train))
print("Performance du modèle linéaire Entrainé : ", r2_score(y_test, y_pred))
xline = np.linspace(0, max(y_test), 100)
yline = xline
plt.plot(xline, yline, color="red")
plt.scatter(y_test, y_pred, color="blue")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted values")
plt.show()
#Modèle polynomial
print("\nModèle polynomial")
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
X_test_poly = poly_reg.transform(X_test)
y_pred = lin_reg_2.predict(X_test_poly)
r2_score(y_test, y_pred)
mean_squared_error(y_test, y_pred)
print("score modèle polynomiale de degré 4:", r2_score(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))



#Arbre de décision
print("\nArbre de décision")
# Initialiser l'arbre de décision
clf = DecisionTreeClassifier()
# Entraîner l'arbre de décision avec des données d'entraînement
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

clf = clf.fit(X_train, y_train)
# Utiliser l'arbre de décision pour prédire des résultats sur des données de test
y_pred = clf.predict(X_test)
# Evaluer la performance de l'arbre de décision
print("score arbre de décision:", r2_score(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))