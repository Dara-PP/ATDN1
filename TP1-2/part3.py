import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configuration pour rendre les données déterministes
np.random.seed(42)

# 1. Génération des données simulées que l'on cree fictivement 
n = 100
surface = np.random.randint(50, 200, n)  # Surface en m²
nb_pieces = np.random.randint(1, 6, n)   # Nombre de pièces
age = np.random.randint(0, 50, n)        # Âge de la maison en années
distance_centre = np.random.uniform(0, 20, n)  # Distance au centre-ville en km

# Prix de la maison
prix = 30 + (surface * 0.8) + (nb_pieces * 15) - (age * 0.5) - (distance_centre * 2) + np.random.normal(0, 10, n)

# Création du DataFrame
data = pd.DataFrame({
    'Surface': surface,
    'Nb_pieces': nb_pieces,
    'Age': age,
    'Distance_centre': distance_centre,
    'Prix': prix
})

# 2. Création de la variable cible binaire
prix_median = np.median(prix)       # On creer la mediane selon le prix 
data['Prix_cible'] = (data['Prix'] > prix_median).astype(int) # On ajoute le astype afin d'avoir des valeur entiere 1 et 0 

# Affichage des premières lignes pour vérifier nos 5 premieres valeurs 
print(data.head(5))

# 3. Diviser les données en ensemble d'entraînement et ensemble de test.
X = data[['Surface', 'Nb_pieces', 'Age', 'Distance_centre']].values
y = data['Prix_cible'].values

# 4. Standardiser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entraîner le modèle de régression logistique
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Prédictions
y_pred = log_reg.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Précision du modèle :", accuracy)
print("Matrice de confusion:", conf_matrix)
print("Rapport de classification:", class_report)

# Affichage de la distribution de la variable cible
plt.figure(figsize=(8, 5))
plt.hist(data['Prix'],  color='orange')
plt.axvline(prix_median, color='red', label= 'Médiane du Prix')
plt.xlabel("Prix de la maison")
plt.ylabel("Nombre de maisons")
plt.title("Distribution des prix des maisons")
plt.legend()
plt.show()
