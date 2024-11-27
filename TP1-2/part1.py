import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from scipy import stats

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score 


# Configuration de Seaborn pour style
sns.set_style("whitegrid")

"""Regression Lineaire"""
#.1 Generation et visualisation des données 

# Génération de données  : 
np.random.seed(0)
X = 2 * np.random.rand(100, 1)          # 100 échantillons pour la variable X indépendantes
bruit = np.random.randn(100, 1)         # Bruit gaussien avec randn bruit realiste
Y = 7 + 4 * X + bruit                   # Variable Y avec bruit variables dependante

# Visualisation des données
plt.figure(figsize=(10, 6))             # Taille figure graphique
plt.scatter(X, Y, label='échantillons') # Trace les variables x,y
# Informations graphiques
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Données générées de base')
plt.legend()
plt.show()

# 2. Régression linéaire 
regression = LinearRegression()         # Fonction de regressions sur scikit-learn
regression.fit(X, Y)
y_pred = regression.predict(X)          

# Visualisation des prédictions
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Données observées')                    # Données observés
plt.plot(X, y_pred, color='red', label='Prédictions du modèle') # Predictions du modele 
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Régression linéaire avec distribution gaussienne')
plt.legend()
plt.show()

# 3. Calcul des résidus
residus = (Y - y_pred).flatten()

# 4. Visualisation Histogramme et Q-Q plot 
# Visualisation des résidus avec un histogramme
plt.figure(figsize=(10, 6))
sns.histplot(residus, kde = True) # Creer l'historigramme avec 
plt.xlabel('Résidus')
plt.ylabel('Densité')
plt.title('Distribution des résidus')
plt.show()

# Utilisation de probplot pour le Q-Q plot
plt.figure(figsize=(10, 6))
stats.probplot(residus, dist="norm", plot=plt)  # Creer une distribution de type normal avec nos residus
plt.title("Q-Q plot Lineaire")
plt.show()

# Régression polynomiale de degré superieur
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
regression_poly = LinearRegression()
regression_poly.fit(X_poly, Y)

X_sorted = np.sort(X, axis=0)                   # Trie X pour un tracé lisse car sinon probleme
X_poly = poly_reg.transform(X_sorted)           # Transformation polynomiale de X trié
y_pred_poly = regression_poly.predict(X_poly)   # Prédictions sur X trié

# Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Échantillons', color='blue')
plt.plot(X_sorted, y_pred_poly, color='green', label='Modèle polynomial')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Régression linéaire et polynomiale')
plt.legend()
plt.show()

# 3. Calcul des résidus
residus_poly = (Y - y_pred_poly).flatten()

# 4. Visualisation Histogramme et Q-Q plot 
# Visualisation des résidus avec un histogramme
plt.figure(figsize=(10, 6))
sns.histplot(residus_poly, kde = True) # Creer l'historigramme avec 
plt.xlabel('Résidus')
plt.ylabel('Densité')
plt.title('Distribution des résidus')
plt.show()

# Utilisation de probplot pour le Q-Q plot
plt.figure(figsize=(10, 6))
stats.probplot(residus_poly, dist="norm", plot=plt)  # Creer une distribution de type normal avec nos residus
plt.title("Q-Q plot polynomiale")
plt.show()


# calcul coefficient de détermination (R2), la fiabilite du modele Lineaire 0.84 ce qui est enormal pour un modele Lineaire
r2 = r2_score(Y,y_pred)    
# Le RMSE est normal 0.9962121504602562 mais pas tres optimise                  
rmse_lin = np.sqrt(mean_squared_error(Y, y_pred))
print(r2)  
print(rmse_lin)

# calcul coefficient de détermination (R2) du modele Polynomiale -1.42 ce qui n'est pas normal pour un modele Polynomiale ....
r2 = r2_score(Y,y_pred_poly)   
# Le RMSE l'erreur quadratique est egalement res mauvaise tres loin des 0 (3.884388170918413).           
rmse_poly = np.sqrt(mean_squared_error(Y, y_pred_poly))
print(r2)
print(rmse_poly) 

coeffs_lin = regression.coef_
coeffs_poly = regression_poly.coef_
print(coeffs_lin)
print(coeffs_poly)
"""
Valeur des coefficients : 
Lineaire [[3.96846751]]
Polynomiale [[ 0.          4.84100842 -0.45190593]]
"""


