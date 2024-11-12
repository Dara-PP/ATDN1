import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

"""I - Régression Linéaire et Polynomiale avec Visualisation et Analyse des Résidus"""

data = pd.read_csv('./house-prices-advanced-regression-techniques/train.csv') # Chargment du CSV Cible 

X = data[['GrLivArea']].values  # Variables explicative surface de terrain
Y = data['SalePrice'].values  # Variable cible prix de vente


"""1. Ajustement du Modèle Linéaire :"""
# Visualisation de la relation entre Surface habitable et Prix de vente
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue')
plt.xlabel('Surface habitable totale (GrLivArea)')
plt.ylabel('Prix de vente (SalePrice)')
plt.title('Surface de Terrain')
plt.legend()
plt.show()

# Regression Lineaire
regression = LinearRegression()
regression.fit(X, Y)
y_pred = regression.predict(X)

# Visualisation des prédictions modèle Linéaire
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', alpha=0.5, label='Données observées')
plt.scatter(X, y_pred, color='red', alpha=0.5, label='Prédictions du modèle linéaire')
plt.xlabel('Surface habitable totale (GrLivArea)')
plt.ylabel('Prix de vente (SalePrice)')
plt.title('Régression Linéaire ')
plt.legend()
plt.show()

# Calcul et visualisation des résidus
residus = Y - y_pred

# Histograme des résidus
plt.figure(figsize=(10, 6))
sns.histplot(residus, kde=True)
plt.xlabel('Résidus')
plt.ylabel('Densité')
plt.title('Distribution des Résidus')
plt.show()

# Q-Q Plot des résidus
plt.figure(figsize=(10, 6))
stats.probplot(residus, dist="norm", plot=plt)
plt.title("Q-Q plot des résidus")
plt.show()

"""2 . Ajustement du Modèle Polynomial :"""

# Régression polynomiale (degré 2)
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
regression_poly = LinearRegression()
regression_poly.fit(X_poly, Y)

X_sorted = np.sort(X, axis=0)                   # Trie X pour un tracé lisse car sinon probleme
X_poly = poly_reg.transform(X_sorted)           # Transformation polynomiale de X trié
y_pred_poly = regression_poly.predict(X_poly)   # Prédictions sur X trié

# Visualisation du modèle polynomial
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue')
plt.scatter(X, y_pred_poly, color='green', alpha=0.5) # Prediction modele Polynomiale
plt.xlabel('Surface habitable totale (GrLivArea)')
plt.ylabel('Prix de vente (SalePrice)')
plt.title('Régression polynomiale ')
plt.legend()
plt.show()

# 3. Régression Polynomiale avec régularisation Ridge (degré 2)
ridge_reg = Ridge(alpha=1.0)  # Choix d'une valeur de régularisation alpha=1.0
ridge_reg.fit(X_poly, Y)
y_pred_ridge = ridge_reg.predict(X_poly)

# Visualisation des prédictions du modèle polynomial avec régularisation
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Données observées')
plt.plot(np.sort(X, axis=0), np.sort(y_pred_poly, axis=0), color='green')
plt.plot(np.sort(X, axis=0), np.sort(y_pred_ridge, axis=0), color='purple')
plt.xlabel('Surface habitable totale (GrLivArea)')
plt.ylabel('Prix de vente (SalePrice)')
plt.title('Comparaison des modèles avec et sans régularisation Ridge')
plt.legend()
plt.show()

# 4. Comparaison des Coefficients

# Calcul de l'erreur quadratique moyenne (RMSE) et du R2 Lineaire 
rmse = np.sqrt(mean_squared_error(Y, y_pred))
r2 = r2_score(Y, y_pred)
print(rmse)             # 56034.303865279944
print(r2)               # 0.5021486502718042

# Calcul de l'erreur quadratique moyenne (RMSE) et du R2 Polynomiale
rmse_poly = np.sqrt(mean_squared_error(Y, y_pred_poly))
r2_poly = r2_score(Y, y_pred_poly)
print(rmse_poly)        # 98483.61544430589
print(r2_poly)          # -0.5378702456449553

# Calcul de l'erreur quadratique moyenne (RMSE) et du R2 Ridge
rmse_ridge = np.sqrt(mean_squared_error(Y, y_pred_ridge))
r2_ridge = r2_score(Y, y_pred_ridge)
print( rmse_ridge)      # 79398.42392700305
print(r2_ridge)         # 0.0004247586899346345

# Coefficient Ridge 
print(regression_poly.coef_) # [ 0.00000000e+00  1.45549389e+02 -1.02504204e-02]
print(ridge_reg.coef_)       # [ 0.00000000e+00 -4.61750912e+00  4.10715169e-04]


"""II - Validation Croisée et Intervalle de Confiance"""
# Validation Croisée :
