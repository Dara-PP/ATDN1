import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, mode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Chargement du CSV 
df = pd.read_csv("rendement_mais.csv", encoding="latin1", sep=",")

# 1. Prétraitement des données
# Test affichage des 5 premières lignes du DataFrame
print("Colonnes du dataset :", df.columns.tolist())

# Nettoyage des données 
df = df.drop_duplicates() # Supprime les doublons 
df = df.dropna(subset=["SURFACE_HA", "TYPE_SOL", "ENGRAIS_KG_HA", "PRECIPITATIONS_MM", "TEMPERATURE_C", "RENDEMENT_T_HA"]) # Supprime les valeurs manquantes

# 2. Analyse des données
# Statistiques descriptives et visualisation
rend = df["RENDEMENT_T_HA"] # Variable cible
# 2.1
print("Moyenne:", rend.mean())
print("Médiane:", rend.median())
print("Mode:", mode(rend)[0][0])
# 2.2
print("Ecart type:", rend.std())
print("Variance:", rend.var())
print("Etendue:", rend.max() - rend.min())

# 3 Visualisation Historigramme Boxplot : 
# Historigramme 2.3
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(rend, edgecolor='black')
plt.title("Historigramme")

# Boxplot 2.3
plt.subplot(1,2,2)
plt.boxplot(rend)
plt.title("Boxplot")
plt.tight_layout()
plt.show()

# 2.4 Heatmap matrice de corrélation
num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()
plt.figure(figsize=(6,5))
plt.imshow(corr)
plt.title("Matrice de corrélation")
plt.show()

# 4. ANOVA
groupe_argileux = df[df["TYPE_SOL"]=="Argileux"]["RENDEMENT_T_HA"]
groupe_sableux  = df[df["TYPE_SOL"]=="Sableux"]["RENDEMENT_T_HA"]
groupe_limoneux = df[df["TYPE_SOL"]=="Limoneux"]["RENDEMENT_T_HA"]
anova = f_oneway(groupe_argileux, groupe_sableux, groupe_limoneux)
print("ANOVA F_value :", anova.statistic, "\np-value :", anova.pvalue)

# 5. Prédiction avec différents modeles 
variables = ["SURFACE_HA", "ENGRAIS_KG_HA", "PRECIPITATIONS_MM", "TEMPERATURE_C"]
X = df[variables]
y = df["RENDEMENT_T_HA"]

# A. LineaireRegression 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% des données de test 80% de train
model = LinearRegression()
model.fit(X_train, y_train)

y_test_pred  = model.predict(X_test)

print("\nPerformance LinearRegression test:")
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("R^2:", r2_score(y_test, y_test_pred))

# B. SVR
svr_model = SVR(kernel='linear', C=1.0, epsilon=0.1)
svr_model.fit(X_train, y_train)

y_pred_svr = svr_model.predict(X_test)

print("\nPerformance SVR test:")
print("MAE:", mean_absolute_error(y_test, y_pred_svr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_svr)))
print("R^2:", r2_score(y_test, y_pred_svr))

# Coefficients du modele
print(f"\nCoefficients du modele [SURFACE_HA,ENGRAIS_KG_HA, PRECIPITATIONS_MM, TEMPERATURE_C] : {model.coef_}")
