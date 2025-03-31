
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def init():
    """Initialisation des données et préparation avec notre fonction d'objectif"""
    # Chargement des données
    data = pd.read_csv("tp2_atdn_donnees.csv", encoding="latin1", sep=",")
    data.columns = ['Humidite_%', 'Temperature_C', 'pH_sol', 'Precipitations_mm', 'Type_sol', 'Rendement_t_ha']

    # Nettoyage & Tests
    data = data.dropna() # Supprime les doublons 
    data = data.dropna(subset=['Humidite_%', 'Temperature_C', 'pH_sol', 'Precipitations_mm', 'Type_sol', 'Rendement_t_ha']) # Supprime les valeurs manquantes
    print("Colonnes du dataset test :", data.columns.tolist()) # Test de la présence des colonnes

    # Fonction d'objectif (rendement agricole)
    X = data[['Humidite_%', 'Temperature_C']].values
    y = data['Rendement_t_ha'].values
    X = data[['Humidite_%', 'Temperature_C', 'pH_sol', 'Precipitations_mm']]
    y = data['Rendement_t_ha']
    return X, y, data

def grid(X, y):
    """Optimisation des hyperparamètres avec GridSearchCV et RandomizedSearchCV"""
    # Définition de l'espace d'hyperparamètres
    param_dist = {'n_estimators': np.arange(10, 201, 10), 'max_depth': np.arange(3, 21, 1)}

    # Optimisation par GridSearchCV
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid=param_dist, cv=5)
    grid_search.fit(X, y)

    # Optimisation par RandomizedSearchCV
    random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist, n_iter=30, cv=5)
    random_search.fit(X, y)
    print("Meilleurs hyperparamètres Grid Search :", grid_search.best_params_)
    print("Meilleurs hyperparamètres Random Search :", random_search.best_params_)

    return grid_search, random_search

def draw(grid_search, random_search):
    """Visualisation des résultats de l'optimisation"""
    # Courbe de convergence
    plt.figure(figsize=(8, 4))
    results_grid = pd.DataFrame(grid_search.cv_results_['mean_test_score'], columns=['Grid Search'])
    results_random = pd.DataFrame(random_search.cv_results_['mean_test_score'], columns=['Random Search'])

    plt.plot(results_grid, label='Grid Search', marker='o')
    plt.plot(results_random, label='Random Search', marker='x')

    plt.xlabel('Itérations')
    plt.ylabel('Score moyen CV')
    plt.title('Courbe de convergence des méthodes d’optimisation')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    X, y, data = init()
    grid_search, random_search = grid(X, y)
    draw(grid_search, random_search)