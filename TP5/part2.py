import time
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


"""Exercice 2 : Ensembles d'apprentissage (Random Forest et Gradient Boosting)"""

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

# 4. Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entraînement du modèle Random Forest
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)  # Hyperparamètres ajustables
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - start_time
print(f"Temps d'entraînement Random Forest: {rf_train_time:.2f} secondes")

# 6. Prédictions et évaluation du Random Forest
y_pred_rf = rf_model.predict(X_test)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]  # Probabilités nécessaires pour la courbe ROC
roc_auc_rf = roc_auc_score(y_test, y_pred_prob_rf)

print("Random Forest - Matrix de confusion:")
print(confusion_matrix(y_test, y_pred_rf))
print("Random Forest - Rapport de classification:")
print(classification_report(y_test, y_pred_rf))

# 7. Entraînement du modèle Gradient Boosting XGBoost
start_time = time.time()
xgb_model = XGBClassifier(n_estimators=100, random_state=42, max_depth=10, use_label_encoder=False, eval_metric='logloss')  # Hyperparamètres ajustables
xgb_model.fit(X_train, y_train)
xgb_train_time = time.time() - start_time
print(f"Temps d'entraînement XGBoost: {xgb_train_time:.2f} secondes")

# 8. Prédictions et évaluation du Gradient Boosting
start_time1 = time.time() # lancement time pour mesure du temps 
y_pred_xgb = xgb_model.predict(X_test)
y_pred_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]  # Probabilités nécessaires pour la courbe ROC
roc_auc_xgb = roc_auc_score(y_test, y_pred_prob_xgb)
train_time1 = time.time() - start_time1
print(f"Temps de prédiction : {train_time1} secondes")

print("XGBoost - Matrix de confusion:")
print(confusion_matrix(y_test, y_pred_xgb))
print("XGBoost - Rapport de classification:")
print(classification_report(y_test, y_pred_xgb))

# 9. Tracer les courbes ROC-AUC pour comparer les modèles
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_prob_xgb)

plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, color='blue', label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_xgb, tpr_xgb, color='green', label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Courbes ROC - Random Forest vs XGBoost')
plt.xlabel('False Positive Rate FPR')
plt.ylabel('True Positive Rate TPR')
plt.legend()
plt.grid()
plt.show()
