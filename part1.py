import time
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import svm 


"""Exercice 1. Classification avec SVM"""
# Chargement du fichier CSV avec le bon encodage
data = pd.read_csv('./sms_spam.csv', encoding='latin1') 

# Prétraitement des données
# Nettoyage des données
data = data[['v1', 'v2']]  # Garder uniquement les colonnes nécessaires pas les Nan sans cette ligne nous avons des Nan qui apparaissent
data = data.dropna(subset=['v1', 'v2'])  # Supprimer les lignes avec des valeurs manquantes
data['v2'] = data['v2'].str.strip()  # Nettoyer les messages

# Convertir les labels en valeurs numériques
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1}) 

# Vectorisation des textes
vectorizer = CountVectorizer(stop_words='english', max_features=3000)  # Limiter à 3000 mots
X = vectorizer.fit_transform(data['v2']).toarray()
y = data['v1'].values

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle SVM 
start_time = time.time() # lancement time pour mesure du temps 
svm_model = svm.SVC(kernel='linear', probability=True) #creation d'une instance du modele SVM
svm_model.fit(X_train, y_train) # lancement de l'entrainement du modele 
train_time = time.time() - start_time
print(f"Temps d'entraînement: {train_time} secondes")

# Prédictions et évaluation
"""  # Le modele prédit apres entrainement sur l'ensemble de test ici X_test y_pred contient ensuite les classes prédites par le modele"""
y_pred = svm_model.predict(X_test)

# Évaluation avec les différentes metrics
print("Rapport de classification:")
print(classification_report(y_test, y_pred))

# Calcul du ROC-AUC
y_pred_prob = svm_model.decision_function(X_test) #Le ROC_AUC necéssite des probabilites ou des scores de décision
roc_auc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Tracer la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray',)
plt.title('Courbe ROC modèle SVM')
plt.xlabel('False Positive Rate FPR')
plt.ylabel('True Positive Rate TPR')
plt.grid()
plt.show()