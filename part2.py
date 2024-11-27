import time
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

from sklearn.feature_extraction.text import CountVectorizer

"""Exercice 2. Mélange de modèles (Voting Classifier)"""
# Chargement des données
data = pd.read_csv('./sms_spam.csv', encoding='latin1')

# Prétraitement
data = data[['v1', 'v2']]
data = data.dropna(subset=['v1', 'v2'])
data['v2'] = data['v2'].str.strip()
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})

# Vectorisation des textes
vectorizer = CountVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(data['v2']).toarray()
y = data['v1'].values

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèles individuels
clf1 = LogisticRegression(max_iter=1000)
clf2 = svm.SVC(kernel='linear', probability=True)
clf3 = MultinomialNB()

# Entraînement des differents modèles
start_time = time.time()
clf1.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Temps d'entraînement (LogisticRegression) : {train_time:.2f} secondes")
start_time = time.time()
clf2.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Temps d'entraînement (SVM) : {train_time:.2f} secondes")
start_time = time.time()
clf3.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Temps d'entraînement (Bayes) : {train_time:.2f} secondes")

"""  # Les modeles prédits apres entrainement sur l'ensemble de test ici X_test y_pred contient ensuite les classes prédites par les modeles"""

y_pred_clf1 = clf1.predict(X_test)
y_pred_clf2 = clf2.predict(X_test)
y_pred_clf3 = clf3.predict(X_test)

# Évaluation avec les différentes metrics
print("Matrix Logistic Recgression:", confusion_matrix(y_test, y_pred_clf1))
print(classification_report(y_test, y_pred_clf1))
print("Matrix SVM:", confusion_matrix(y_test, y_pred_clf2))
print(classification_report(y_test, y_pred_clf2))
print("Naives Bayes:", confusion_matrix(y_test, y_pred_clf3))
print(classification_report(y_test, y_pred_clf3))

# Courbe ROC pour Logistic Regression
y_pred_prob1 = clf1.predict_proba(X_test)[:, 1]
fpr1, tpr1, _ = roc_curve(y_test, y_pred_prob1)
roc_auc1 = roc_auc_score(y_test, y_pred_prob1)

# Courbe ROC pour SVM
y_pred_prob2 = clf2.predict_proba(X_test)[:, 1]
fpr2, tpr2, _ = roc_curve(y_test, y_pred_prob2)
roc_auc2 = roc_auc_score(y_test, y_pred_prob2)

# Courbe ROC pour Naive Bayes
y_pred_prob3 = clf3.predict_proba(X_test)[:, 1]
fpr3, tpr3, _ = roc_curve(y_test, y_pred_prob3)
roc_auc3 = roc_auc_score(y_test, y_pred_prob3)

# Courbe ROC pour Voting Classifier (Hard)
hard_voting = VotingClassifier(estimators=[
    ('lr', clf1), ('svm', clf2), ('nb', clf3)
], voting='hard')
start_time = time.time()
hard_voting.fit(X_train, y_train)
y_pred_hard = hard_voting.predict(X_test)
train_time = time.time() - start_time
print(f"Temps d'entraînement (Classifier Hard) : {train_time:.2f} secondes")

# Courbe ROC pour Voting Classifier (Soft)
soft_voting = VotingClassifier(estimators=[
    ('lr', clf1), ('svm', clf2), ('nb', clf3)
], voting='soft')
start_time = time.time()
soft_voting.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Temps d'entraînement (Classifier Soft) : {train_time:.2f} secondes")
y_pred_prob_voting = soft_voting.predict_proba(X_test)[:, 1]
fpr_v, tpr_v, _ = roc_curve(y_test, y_pred_prob_voting)
roc_auc_v = roc_auc_score(y_test, y_pred_prob_voting)

# Convertir les probabilités en classes binaires
y_pred_binary1 = (y_pred_prob_voting >= 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred_binary1)
print("Classifier SOFT:", confusion_matrix(y_test, y_pred_binary1))
print(classification_report(y_test, y_pred_binary1))

# Tracer du Graphique
plt.figure(figsize=(8, 8))
plt.plot(fpr1, tpr1, label=f'Logistic Regression (AUC = {roc_auc1:.2f})', color='blue')
plt.plot(fpr2, tpr2, label=f'SVM (AUC = {roc_auc2:.2f})', color='green')
plt.plot(fpr3, tpr3, label=f'Naive Bayes (AUC = {roc_auc3:.2f})', color='red')
plt.plot(fpr_v, tpr_v, label=f'Voting Classifier Soft (AUC = {roc_auc_v:.2f})', color='purple')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.title("Courbes ROC de tous les modèles")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()
