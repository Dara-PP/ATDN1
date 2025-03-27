import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
from scipy.stats import zscore as sp_zscore, shapiro, f_oneway
import scipy.stats as stats

# Liste des variables à tester pour ce TP
variables = ["Poids_poulet_g", "Nourriture_consommee_g_jour", "Temperature_enclos_C"]
noms = ["Poids", "Nourriture", "Température"]

def init():
    # Chargement du CSV avec le bon encodage et séparateur
    df = pd.read_csv("donnees_elevage_poulet.csv", encoding="latin1", sep=",")
    # 1. Prétraitement des données
    # Test affichage des colones du DataFrame
    print("Colonnes du dataset :", df.columns.tolist())
    # Nettoyage des données 
    df = df.drop_duplicates() # Supprime les doublons 
    df = df.dropna(subset=['Poids_poulet_g', 'Nourriture_consommee_g_jour', 'Temperature_enclos_C', 'Humidite_%', 'Age_poulet_jours', 'Gain_poids_jour_g', 'Taux_survie_%', 'Cout_elevage_FCFA']) # Supprime les valeurs manquantes

    # Test log transformée pour les outliers sur des données non normal
    df["Poids_poulet_g_log"] = np.log(df["Poids_poulet_g"])
    df["Nourritue_poulet_g_log"] = np.log(df["Nourriture_consommee_g_jour"])

    return df

# 2. Analyse des données
# Statistiques descriptives et visualisation
# Exercice 1
# Variable Nourriture
def draw():
    bouf = df["Nourriture_consommee_g_jour"] 
    print("\nCalcul Nourriture")
    print("Moyenne:", bouf.mean())
    print("Médiane:", bouf.median())
    print("Ecart-type:", bouf.std())
    print("Variance:", bouf.var())
    print("Quartiles:", bouf.quantile([0.25, 0.5, 0.75]).to_dict())
    print("Etendue:", bouf.max() - bouf.min())

    # Variable poids
    print("\nCalcul poids")
    poids = df["Poids_poulet_g"] 
    print("Moyenne:", poids.mean())
    print("Médiane:", poids.median())
    print("Ecart-type:", poids.std())
    print("Variance:", poids.var())
    print("Quartiles:", poids.quantile([0.25, 0.5, 0.75]).to_dict())
    print("Etendue:", poids.max() - poids.min())

    # Variable temp
    print("\nCalcul Temperature")
    temp = df["Temperature_enclos_C"] 
    print("Moyenne:", temp.mean())
    print("Médiane:", temp.median())
    print("Ecart-type:", temp.std())
    print("Variance:", temp.var())
    print("Quartiles:", temp.quantile([0.25, 0.5, 0.75]).to_dict())
    print("Etendue:", temp.max() - temp.min())

    # 3. Visualisation historigramme/boxtplot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # Premiere ligne : historigrammes
    axes[0, 0].hist(bouf, edgecolor='black')
    axes[0, 0].set_title("Historigramme Nourriture")
    axes[0, 1].hist(poids, edgecolor='black')
    axes[0, 1].set_title("Historigramme Poids")
    axes[0, 2].hist(temp, edgecolor='black')
    axes[0, 2].set_title("Historigramme Température")

    # Deuxieme ligne : boxplots
    axes[1, 0].boxplot(bouf)
    axes[1, 0].set_title("Boxplot Nourriture")
    axes[1, 1].boxplot(poids)
    axes[1, 1].set_title("Boxplot Poids")
    axes[1, 2].boxplot(temp)
    axes[1, 2].set_title("Boxplot Température")
    plt.tight_layout()
    plt.show()

    # Historigramme test log
    plt.hist(df["Poids_poulet_g_log"], bins=30, edgecolor='black')
    plt.title("Histogramme de Poids_poulet_g (log)")
    plt.show()

# Exercice 2 
def plot_qq_simple(df, column):
    """
    QQ-plot comparer les quantiles à ceux d'une loi normale. 
    (Check pour outliers)
    """
    plt.figure(figsize=(6, 4))
    stats.probplot(df[column], dist="norm", plot=plt)
    plt.title(f"QQ Plot - {column}")
    plt.xlabel("Quantiles théoriques")
    plt.ylabel("Quantiles observés")
    plt.show()

# IQR
def iqr(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr_val = q75 - q25
    Q1= q25 - 1.5 * iqr_val # 1.5 Valeur proche distribution gaussienne
    Q3 = q75 + 1.5 * iqr_val
    clean = data[(data >= Q1) & (data <= Q3)]
    outliers = data[(data < Q1) | (data > Q3)]
    return clean, outliers

# Z_score
def zscore(data, threshold=2.5): # min du treshhold 2.5 max 3
    z_scores = np.abs(sp_zscore(data))
    clean = data[z_scores <= threshold]
    outliers = data[z_scores > threshold]
    return clean, outliers

def outlier(df, variables, detection_func, color, method_name):
    for var in variables:
        clean, outliers = detection_func(df[var])
        print(f"Variable: {var}")
        print("Clean data:")
        print(clean)
        print("Outliers:")
        print(outliers)
        
        plt.figure(figsize=(6,4))
        plt.boxplot(df[var], showfliers=False)
        plt.scatter(np.ones(len(outliers)), outliers, color=color, label="Outliers")
        plt.title(f"Boxplot de {var} ({method_name})")
        plt.ylabel(var)
        plt.legend()
        plt.show()

# Exercice 3 
def test_normalité(data, variable_name):
    """Teste la normalité avec Shapiro-Wilk"""
    stat, p = shapiro(data)
    if p > 0.05:
        print(f"La variable {variable_name} suit une loi normale (p = {p:.3f}).")
    else:
        print(f"La variable {variable_name} ne suit pas une loi normale (p = {p:.3f}).")

def anova(df, target, group_var):
    """Test ANOVA pour pour comparer les moyennes de plusieurs groupes"""
    group1 = df[df[group_var] == "Jeune"][target].values
    group2 = df[df[group_var] == "Moyen"][target].values
    group3 = df[df[group_var] == "Vieux"][target].values
    # test ANOVA
    stat, p = f_oneway(group1, group2, group3)
    print(f"Comparaison de la moyenne de '{target}' selon les groupes de '{group_var}'")
    print(f"Statistique F : {stat:.3f}")
    print(f"Valeur p : {p:.3f}")
    if p < 0.05:
        print("Différence significative entre les groupes.")
    else:
        print("Pas de différence significative entre les groupes.")

def assign_age_group(age):
    if age <= 10:
        return "Jeune"
    elif age <= 20:
        return "Moyen"
    else:
        return "Vieux"

def assign_group(age):
    if age <= median_age:
        return "Groupe1"
    else:
        return "Groupe2"
    
if __name__ == "__main__":

    # Classic
    df = init()
    draw()

    # Visualisation normal "bonus"
    plot_qq_simple(df, "Poids_poulet_g")
    plot_qq_simple(df, "Poids_poulet_g_log")  

    # Marche pas trop bien ! 
    """# Affichage avec la méthode IQR (rouge)
    outlier(df, variables, iqr, color="red", method_name="IQR")
    # Affichage avec la méthode IQR (rouge)
    outlier(df, ["Poids_poulet_g_log"], iqr, color="purple", method_name="IQR (log)")
    # Affichage avec la méthode Z-Score (bleu)
    outlier(df, variables, zscore, color="blue", method_name="Z-Score")
    # Affichage avec la méthode Z-Score (violet)
    """
    outlier(df, ["Poids_poulet_g_log"], zscore, color="purple", method_name="Z-Score (log)")

    # test normalité 
    for i in range(len(variables)):
        test_normalité(df[variables[i]], noms[i])
    test_normalité(df["Poids_poulet_g_log"],noms[0])

    # Preparation pour test t avec l'age 
    df["Age_group"] = df["Age_poulet_jours"].apply(assign_age_group)
    anova(df, "Poids_poulet_g", "Age_group")
    median_age = df["Age_poulet_jours"].median()
    df["Age_group_t"] = df["Age_poulet_jours"].apply(assign_group)
    
    # test t de Student pour comparer Poids_poulet_g entre Groupe1 et Groupe2
    groupe1 = df[df["Age_group_t"] == "Groupe1"]["Poids_poulet_g"]
    groupe2 = df[df["Age_group_t"] == "Groupe2"]["Poids_poulet_g"]
    t_stat, p_value = ttest_ind(groupe1, groupe2)
    
    print("Test t de Student pour Poids_poulet_g entre Groupe1 et Groupe2 :")
    print("t-statistique =", t_stat)
    print("p-value =", p_value)