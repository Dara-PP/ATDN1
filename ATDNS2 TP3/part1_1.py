import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as k

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

def baye(X, y):
    """Optimisation bayésienne avec un modèle GP"""
    # Définition du modèle GP
    kernel = k(1.0) * RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42) # Noyau RBF
    # Ajustement du modèle GP
    gp.fit(X, y)
    # Prédiction sur grille de points
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
    x1x2 = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
    y_pred, sigma = gp.predict(x1x2, return_std=True)
    return y_pred, sigma

def draw(X, y):
    """Visualisation des résultats"""
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label="Points évalués")
    plt.colorbar(sc, label="Rendement agricole (t/ha)")
    plt.xlabel('Humidité (%)')
    plt.ylabel('Température (°C)')
    plt.title('Optimisation Bayésienne sur la Production Agricole')
    plt.show()

if __name__ == '__main__':
    """Initialisation des données et exécution de l'optimisation bayésienne"""
    X, y, data = init()
    y_pred, sigma = baye(X, y)
    draw(X, y)

