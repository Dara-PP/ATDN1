import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB  # Naive Bayes

"""Exercice 1. Classification avec SVM"""
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
vectorizer = CountVectorizer(stop_words='english', max_features=3000)  # Limiter à 3000 mots
X = vectorizer.fit_transform(data['v2']).toarray()
y = data['v1'].values
print(data['v1'][5])

# 4. Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entraîner le modèle Naive Bayes
start_time = time.time()
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Temps d'entraînement: {train_time} secondes")

# 6. Prédictions et évaluation
y_pred = nb_model.predict(X_test)  # Prédire sur l'ensemble de test

# Évaluation
accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
recallsScore = recall_score(y_test, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Afficher les résultats
print("Metrique de précision :", accuracy)
print("Score F1 :", f1score)
print("Score Recall :", recallsScore)



print("Matrice de confusion:")
print(conf_matrix)
print("Rapport de classification:")
print(class_report)