import time
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

"""Exercice 4. Analyse des modèles avec SHAP"""

# 1. Chargement du fichier CSV avec le bon encodage
data = pd.read_csv('./sms_spam.csv', encoding='latin1') 

# 2. Prétraitement des données
# Nettoyage des données
data = data[['v1', 'v2']]  # Garder uniquement les colonnes nécessaires pas les Nan sans cette ligne nous avons des Nan qui apparaissent
data = data.dropna(subset=['v1', 'v2'])  # Supprimer les lignes avec des valeurs manquantes
# Nettoyage avancé des messages
data['v2'] = data['v2'].str.replace(r'\W', ' ', regex=True)  # Supprimer caractères spéciaux
data['v2'] = data['v2'].str.replace(r'\d+', '', regex=True)  # Supprimer les nombres
data['v2'] = data['v2'].str.replace(r'\s+', ' ', regex=True) # Supprime les espaces multiples
data['v2'] = data['v2'].str.lower()  # Convertir en minuscules
data['v2'] = data['v2'].str.strip()  # Enlever espaces en début/fin


# Convertir les labels en valeurs numériques
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1}) 

# 3. Vectorisation des textes
vectorizer = CountVectorizer(stop_words= 'english',max_features=3000)  # Limiter à 3000 mots
X = vectorizer.fit_transform(data['v2']).toarray()

print(vectorizer.get_feature_names_out())

y = data['v1'].values

# 4. Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entraîner le modèle Random Forest
start_time = time.time() # lancement time pour mesure du temps 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Temps d'entraînement: {train_time} secondes")

# 6. Prédictions et évaluation
start_time1 = time.time() # lancement time pour mesure du temps 
y_pred = rf_model.predict(X_test)
train_time1 = time.time() - start_time1
print(f"Temps de prédiction : {train_time1} secondes")

print("Matrix de confusion:")
print(confusion_matrix(y_test, y_pred))
print("Rapport de classification:")
print(classification_report(y_test, y_pred))

# 7. Application de SHAP
# Création d'un explainer SHAP pour le modèle Random Forest
explainer = shap.Explainer(rf_model, X_train)

# Calcul des valeurs SHAP pour les données de test
shap_values = explainer(X_test)

# 8. Visualisations SHAP
# Diagramme résumé des caractéristiques les plus influentes
shap.summary_plot(shap_values, X_test, feature_names=vectorizer.get_feature_names_out())

# Diagramme de force 
shap.force_plot(
    explainer.expected_value[1],  # Valeur de base pour la classe 1 (spam)
    shap_values[1].values[0],     # Valeurs SHAP pour l'échantillon 0 de la classe 1
    X_test[0],                    # Données brutes pour l'échantillon 0
    matplotlib=True               # Option pour afficher avec matplotlib
)

# Diagramme de dépendance pour une caractéristique spécifique
shap.dependence_plot('free', shap_values.values, X_test, feature_names=vectorizer.get_feature_names_out())
