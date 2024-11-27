import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

"""Exercice 2. Mélange de modèles (Voting Classifier)"""

# Chargement des données
data = pd.read_csv('./sms_spam.csv', encoding='latin1')

# Prétraitement
data = data[['v1', 'v2']]
data = data.dropna(subset=['v1', 'v2'])
data['v2'] = data['v2'].str.strip()
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})

# Vectorisation
vectorizer = CountVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(data['v2']).toarray()
y = data['v1'].values

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Entraînement du GMM
gmm = GaussianMixture(n_components=2, random_state=42)

start_time = time.time()
gmm.fit(X_train)  # Entraînement sur toutes les données
train_time = time.time() - start_time
print(f"Temps d'entraînement (GMM) : {train_time:.2f} secondes")

# 2. Prédictions
# Probabilités d'appartenance pour chaque classe
scores = gmm.predict_proba(X_test)

# Prédire en choisissant la classe avec la probabilité maximale
y_pred = np.argmax(scores, axis=1)

# 3. Évaluation
print("Matrix GMM:", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4. Tracer les frontières de décision
# Réduction des dimensions avec PCA pour visualisation en 2D
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train)
X_test_2D = pca.transform(X_test)

# Réentraîner le GMM sur les données réduites
gmm.fit(X_train_2D)

# Création d'une grille fixe pour tracer les frontières
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Prédiction des classes sur la grille
grid_labels = gmm.predict(grid).reshape(xx.shape)

# Tracé des données et des frontières
plt.figure(figsize=(10, 8))
plt.scatter(X_test_2D[y_test == 0][:, 0], X_test_2D[y_test == 0][:, 1], color='blue', label='Ham')
plt.scatter(X_test_2D[y_test == 1][:, 0], X_test_2D[y_test == 1][:, 1], color='red', label='Spam')
plt.contour(xx, yy, grid_labels, levels=[0.5], cmap="Greys", alpha=0.8)
plt.title("Frontière de décision - GMM")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.legend()
plt.grid()
plt.show()

