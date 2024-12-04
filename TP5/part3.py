import time
import numpy as np
import pandas as pd
from hmmlearn.hmm import MultinomialHMM, CategoricalHMM
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

"""Exercice 3 : Modèles probabilistes avancés (Hidden Markov Models)"""

# 1. Chargement du fichier CSV avec le bon encodage
data = pd.read_csv('./sms_spam.csv', encoding='latin1') 

# 2. Prétraitement des données
# Nettoyage des données
data = data[['v1', 'v2']]  # Garder uniquement les colonnes nécessaires pas les Nan sans cette ligne nous avons des Nan qui apparaissent
data = data.dropna(subset=['v1', 'v2'])  # Supprimer les lignes avec des valeurs manquantes
data['v2'] = data['v2'].str.strip()  # Nettoyer les messages

# Convertir les labels en valeurs numériques
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1}) 

# 3. Vectorisation des textes
vectorizer = CountVectorizer(stop_words='english', max_features=100)  # Limiter à 100 mots
X = vectorizer.fit_transform(data['v2']).toarray()
y = data['v1'].values

# 4. Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 5. Ajustement des données pour HMM en séquence
train_lengths = [len(X_train)]  # Longueur de la séquence d'entraînement
test_lengths = [len(X_test)]  # Longueur de la séquence de test

# 6. Entraînement du modèle HMM
start_time = time.time() # lancement time pour mesure du temps 
hmm_model = MultinomialHMM(n_components=2, n_iter=50, random_state=42, verbose=True)
hmm_model.fit(X_train)  # Entraîner sur les données d'entraînement
train_time = time.time() - start_time
print(f"Temps d'entraînement: {train_time} secondes")

# 7. Prédictions
start_time1 = time.time() # lancement time pour mesure du temps 
y_pred_states = hmm_model.predict(X_test)
y_pred = (y_pred_states == 1).astype(int)  
train_time1 = time.time() - start_time1
print(f"Temps de prédiction : {train_time1} secondes")

# 8. Évaluation des performances
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Courbe ROC
roc_auc = roc_auc_score(y_test, y_pred_states)  
fpr, tpr, _ = roc_curve(y_test, y_pred_states)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Courbe ROC - HMM (Nettoyage des données)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()