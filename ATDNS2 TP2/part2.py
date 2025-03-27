import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA

def init():
    # Chargement du CSV avec le bon encodage et séparateur
    df = pd.read_csv("donnees_elevage_poulet.csv", encoding="latin1", sep=",")
    # 1. Prétraitement des données
    # Test affichage des colones du DataFrame
    print("Colonnes du dataset :", df.columns.tolist())
    # Nettoyage des données 
    df = df.drop_duplicates() # Supprime les doublons 
    df = df.dropna(subset=['Poids_poulet_g', 'Nourriture_consommee_g_jour', 'Temperature_enclos_C', 'Humidite_%', 'Age_poulet_jours', 'Gain_poids_jour_g', 'Taux_survie_%', 'Cout_elevage_FCFA']) # Supprime les valeurs manquantes
    # Test log transformée pour les outliers sur des données non normal
    df["Poids_poulet_g_log"] = np.log(df["Poids_poulet_g"])
    df["Nourritue_poulet_g_log"] = np.log(df["Nourriture_consommee_g_jour"])

    return df

def draw(X): 
  # Visualisation de la projection sur les 2 premières composantes principales
  plt.scatter(proj[:, 0], proj[:, 1], alpha=0.5)
  plt.xlabel("CP1")
  plt.ylabel("CP2")
  plt.title("Projection sur 2 composantes principales")
  plt.show()

  # Matrice de covariance
  plt.figure(figsize=(6,5))
  plt.imshow(matrix, cmap='coolwarm', aspect='auto')
  plt.colorbar() 
  plt.title("Matrice")
  plt.show()

  # Matrice de covariance
  plt.figure(figsize=(6,5))
  plt.imshow(eig_vecs, cmap='coolwarm', aspect='auto')
  plt.colorbar() 
  plt.title("Vecteurs propres")
  plt.show()

  # Test avec ACP et KernelPCA
  proj_pca = ACP_proj(X)
  proj_lin = kPCA(X, 'linear')
  proj_rbf = kPCA(X, 'rbf', gamma=0.01) 
  proj_poly = kPCA(X, 'poly', degree=3)

  # Visualisation des résultats
  fig, axs = plt.subplots(2, 2, figsize=(10, 10))
  axs[0, 0].scatter(proj_pca[:, 0], proj_pca[:, 1], alpha=0.5)
  axs[0, 0].set_title("ACP classique")
  axs[0, 1].scatter(proj_lin[:, 0], proj_lin[:, 1], alpha=0.5)
  axs[0, 1].set_title("KernelPCA linéaire")
  axs[1, 0].scatter(proj_rbf[:, 0], proj_rbf[:, 1], alpha=0.5)
  axs[1, 0].set_title("KernelPCA RBF")
  axs[1, 1].scatter(proj_poly[:, 0], proj_poly[:, 1], alpha=0.5)
  axs[1, 1].set_title("KernelPCA polynomial")
  plt.tight_layout()
  plt.title("Comparaison de l'ACP et du KernelPCA")
  plt.show()

# Exercice 4 
def acp_numpy(X, n_components):
    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    cov_matrix = np.cov(X_norm, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    projection = np.dot(X_norm, eigenvectors[:, :n_components])
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    return cov_matrix, projection, eigenvalues, eigenvectors, explained_variance_ratio

def kPCA(X, kernel, **param):
    return KernelPCA(n_components=2, kernel=kernel, **param).fit_transform(X)

def ACP_proj(X):
    return PCA(n_components=2).fit_transform(X)

if __name__ == "__main__":
  df = init()
  X = df.select_dtypes(include=[np.number]).values
  matrix, proj, eig_vals, eig_vecs, var_explained = acp_numpy(X, n_components=2)

  print("Valeurs propres :", eig_vals)
  print("Variance CP1 et CP2 :", var_explained[:2])
  draw(X)

