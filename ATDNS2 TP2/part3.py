import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

variables= [
    'Poids_poulet_g', 
    'Nourriture_consommee_g_jour', 
    'Temperature_enclos_C', 
    'Humidite_%', 
    'Age_poulet_jours', 
    'Gain_poids_jour_g', 
    'Cout_elevage_FCFA'
]

def init():
    # Chargement du CSV avec le bon encodage et séparateur
    df = pd.read_csv("donnees_elevage_poulet.csv", encoding="latin1", sep=",")
    # 1. Prétraitement des données
    # Test affichage des colones du DataFrame
    print("Colonnes du dataset :", df.columns.tolist())
    # Nettoyage des données 
    df = df.drop_duplicates() # Supprime les doublons 
    df = df.dropna(subset=['Poids_poulet_g', 'Nourriture_consommee_g_jour', 'Temperature_enclos_C', 'Humidite_%', 'Age_poulet_jours', 'Gain_poids_jour_g', 'Taux_survie_%', 'Cout_elevage_FCFA']) # Supprime les valeurs manquantes

    taux_median = df['Taux_survie_%'].median()
    df['Survie'] = (df['Taux_survie_%'] > taux_median).astype(int)
    return df

def evoModel(y_test, y_pred): 
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy : {acc:.2f}")
    print(f"F1-score : {f1:.2f}")
    print("\nRapport de classification :\n", classification_report(y_test, y_pred))
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))

def trainMod(df):
    X = df[variables]
    y = df['Survie']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred, y_test, clf, X_test

def boosting_models(df):
    X_boost = df[variables]
    y_boost = df['Gain_poids_jour_g']
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_boost, y_boost, test_size=0.3, random_state=42)

    ada_reg = AdaBoostRegressor(n_estimators=100, random_state=42)
    ada_reg.fit(X_train_b, y_train_b)
    y_pred_ada = ada_reg.predict(X_test_b)

    gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_reg.fit(X_train_b, y_train_b)
    y_pred_gb = gb_reg.predict(X_test_b)

    # AdaBoost
    print("\nAdaBoost :")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_b, y_pred_ada)):.2f}")
    print(f"MAE: {mean_absolute_error(y_test_b, y_pred_ada):.2f}")
    print(f"R^2: {r2_score(y_test_b, y_pred_ada):.2f}")

    # Gradient Boosting
    print("\nGradient Boosting :")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_b, y_pred_gb)):.2f}")
    print(f"MAE: {mean_absolute_error(y_test_b, y_pred_gb):.2f}")
    print(f"R^2: {r2_score(y_test_b, y_pred_gb):.2f}")

    plt.figure(figsize=(10,5))
    plt.boxplot([y_test_b, y_pred_ada, y_pred_gb], tick_labels=['Réel', 'AdaBoost', 'Gradient Boosting'])
    plt.title("Comparaison des prédictions avec présence d'outliers")
    plt.ylabel('Gain poids par jour (g)')
    plt.show()
    
if __name__ == "__main__":
    df = init()
    y_pred, y_test, clf, X_test = trainMod(df)
    evoModel(y_test, y_pred)
    boosting_models(df)
