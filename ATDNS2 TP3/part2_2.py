import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def init():
    """Initialisation des données et préparation avec notre fonction d'objectif"""
    # Chargement des données
    data = pd.read_csv("tp2_atdn_donnees.csv", encoding="latin1", sep=",")
    data.columns = ['Humidite_%', 'Temperature_C', 'pH_sol', 'Precipitations_mm', 'Type_sol', 'Rendement_t_ha']
    data = data.dropna()
    # Nettoyage & tests
    data = data.dropna() # Supprime les doublons 
    data = data.dropna(subset=['Humidite_%', 'Temperature_C', 'pH_sol', 'Precipitations_mm', 'Type_sol', 'Rendement_t_ha']) # Supprime les valeurs manquantes
    print("Colonnes du dataset test :", data.columns.tolist()) # Test de la présence des colonnes
    # Fonction d'objectif (rendement agricole)
    X = data[['Humidite_%', 'Temperature_C', 'pH_sol', 'Precipitations_mm']]
    y = data['Type_sol']
    return X, y, data

def classBaye(X_train, y_train, X_test):
    """ Classification bayésienne avec GaussianNB. """
    # Classification bayésienne
    bayes_model = GaussianNB()
    bayes_model.fit(X_train, y_train)
    y_pred_bayes = bayes_model.predict(X_test)
    return y_pred_bayes

def classSVM(X_train, y_train, X_test):
    # Classification SVM classique
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    return y_pred_svm

if __name__ == '__main__':
    X, y, data = init()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_pred_bayes = classBaye(X_train, y_train, X_test)
    y_pred_svm = classSVM(X_train, y_train, X_test)

    # Résultats classification
    print("Résultats Classification Bayésienne :\n", classification_report(y_test, y_pred_bayes))
    print("Résultats Classification SVM :\n", classification_report(y_test, y_pred_svm))

