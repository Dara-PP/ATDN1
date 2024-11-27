import time
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import CountVectorizer

"""Exercice 4 : Optimisation des hyperparamètres"""

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

# 1. Optimisation SVM avec GridSearchCV, les parametres C et le kernel du SVM
parameters_svm = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],  
    'gamma': ['scale', 0.01, 0.1] 
}
svm_model = svm.SVC(probability=True)
svm_grid = GridSearchCV(svm_model, parameters_svm, cv=5, scoring='accuracy', verbose=1)

start_time = time.time()
svm_grid.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Temps d'entraînement SVM: {train_time:.2f} secondes")
print("Meilleurs paramètres SVM :", svm_grid.best_params_)

# Prédictions
y_pred_svm = svm_grid.best_estimator_.predict(X_test)

# Courbe ROC
y_pred_prob_svm = svm_grid.best_estimator_.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob_svm)
roc_auc = roc_auc_score(y_test, y_pred_prob_svm)

# Tracer du Graphique 
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC SVM (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Courbe ROC - SVM optimisé")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

# 2. Optimisation GMM 
parameters_gmm = {
    'n_components': [1, 2, 3, 4], # Nombre de composantes du mélange gaussien
    'covariance_type': ['full', 'tied', 'diag', 'spherical'] # Type de matrice de covariance pour ajuster la forme des cluster
}
gmm = GaussianMixture(random_state=42)
gmm_grid = GridSearchCV(gmm, parameters_gmm, cv=5, verbose=1) # Test toutes les combinaisons spécifié dans parameters_gmm

# Entrainement sur les données 
gmm_grid.fit(X_train, y_train)
# Affiches les meilleurs parametres trouvé apres tests
print("Meilleurs paramètres GMM :", gmm_grid.best_params_)

# 3. Optimisation Voting Classifier (ajustement des poids)
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('svm', svm_grid.best_estimator_),
        ('nb', MultinomialNB())
    ],
    voting='soft',
    weights=[1, 2, 3]
)
voting_clf.fit(X_train, y_train)

# Évaluation Voting Classifier
y_pred_voting = voting_clf.predict(X_test)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print("Voting Classifier Accuracy :", accuracy_voting)
