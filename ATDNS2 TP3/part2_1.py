import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel as k


def init():
    """Initialisation des données et préparation avec notre fonction d'objectif"""
    # Chargement des données
    data = pd.read_csv("tp2_atdn_donnees.csv", encoding="latin1", sep=",")
    data.columns = ['Humidite_%', 'Temperature_C', 'pH_sol', 'Precipitations_mm', 'Type_sol', 'Rendement_t_ha']
    # Nettoyage & tests
    data = data.dropna() # Supprime les doublons 
    data = data.dropna(subset=['Humidite_%', 'Temperature_C', 'pH_sol', 'Precipitations_mm', 'Type_sol', 'Rendement_t_ha']) # Supprime les valeurs manquantes
    print("Colonnes du dataset test :", data.columns.tolist()) # Test de la présence des colonnes
    # Fonction d'objectif, en fonction de la température et de l'humidité
    X = data[['Humidite_%', 'Temperature_C']].values
    y = data['Rendement_t_ha'].values
    return X, y, data

def gaussKernel(X, y):
    """ Régression avec noyau gaussien (RBF) """
    # Modèle GP avec noyau RBF
    gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.1)
    gp.fit(X, y)
    # Prédiction avec intervalles de confiance
    y_pred, sigma = gp.predict(X, return_std=True)
    return y_pred, sigma

def test_differents_noyaux(X, y):
    """Fonction de test avec différents noyaux"""
    # Noyau linéaire
    kernel_lin = k(1.0) * DotProduct()
    # Noyau RBF
    kernel_rbf = k(1.0) * RBF(length_scale=1.0)
    # Noyau polynomial degré 3
    kernel_poly = k(1.0) * (DotProduct() ** 3)
    # Création et entrainement des modèles GP
    gp_lin = GaussianProcessRegressor(kernel=kernel_lin, alpha=0.1).fit(X, y)
    gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, alpha=0.1).fit(X, y)
    gp_poly = GaussianProcessRegressor(kernel=kernel_poly, alpha=0.1).fit(X, y)
    # Affichage rapide des scores
    print("Score noyau linéaire:", gp_lin.score(X, y))
    print("Score noyau RBF:", gp_rbf.score(X, y))
    print("Score noyau polynomial:", gp_poly.score(X, y))

def draw(y_pred, sigma, y):
    """ Visualisation des résultats """
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y)), y, color='blue', label='Observé')
    plt.plot(range(len(y_pred)), y_pred, color='red', label='Prédictions')
    plt.fill_between(range(len(y_pred)), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, label='Intervalle confiance 95%')
    plt.xlabel('Échantillons')
    plt.ylabel('Rendement (t/ha)')
    plt.title('Régression bayésienne à noyau')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    X, y, data = init()
    y_pred, sigma = gaussKernel(X, y)
    draw(y_pred, sigma, y)
    test_differents_noyaux(X, y)
