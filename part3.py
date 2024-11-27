import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

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

# 1. Entraînement des GMM (un GMM par classe)
gmm_ham = GaussianMixture(n_components=2, random_state=42)
gmm_spam = GaussianMixture(n_components=2, random_state=42)

start_time = time.time()
# Chaque modele sera focus sur une type de classe
gmm_ham.fit(X_train[y_train == 0])
gmm_spam.fit(X_train[y_train == 1])
train_time = time.time() - start_time
print(f"Temps d'entraînement (GMM) : {train_time:.2f} secondes")

# 2. Prédictions
# Calcul des scores pour chaque classe
scores_ham = gmm_ham.score_samples(X_test)
scores_spam = gmm_spam.score_samples(X_test)

# Prédire en choisissant la classe avec le score le plus élevé
y_pred = (scores_spam > scores_ham).astype(int)

# 3. Évaluation
print(classification_report(y_test, y_pred))

# 4. Tracer les frontières de décision
# Réduction des dimensions avec PCA pour visualisation en 2D
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train)
X_test_2D = pca.transform(X_test)

# Réentraîner les GMM sur les données réduites 
gmm_ham.fit(X_train_2D[y_train == 0])
gmm_spam.fit(X_train_2D[y_train == 1])

# Grille pour tracer les frontières
x_min, x_max = X_train_2D[:, 0].min() - 1, X_train_2D[:, 0].max() + 1
y_min, y_max = X_train_2D[:, 1].min() - 1, X_train_2D[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Scores pour chaque point de la grille
scores_ham_grid = gmm_ham.score_samples(grid)
scores_spam_grid = gmm_spam.score_samples(grid)

# Prédiction sur la grille
decision_boundary = (scores_spam_grid > scores_ham_grid).reshape(xx.shape)

# Tracé des données et des frontières
plt.figure(figsize=(10, 8))
plt.scatter(X_test_2D[y_test == 0][:, 0], X_test_2D[y_test == 0][:, 1], color='blue', label='Ham')
plt.scatter(X_test_2D[y_test == 1][:, 0], X_test_2D[y_test == 1][:, 1], color='red', label='Spam')
plt.contour(xx, yy, decision_boundary, levels=[0.5], cmap="Greys", alpha=0.8)
plt.title("Frontière de décision - GMM")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.legend()
plt.grid()
plt.show()
