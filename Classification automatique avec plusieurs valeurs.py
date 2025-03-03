import pandas as pd
import numpy as np
import re

# Charger les données
file_path = "C:/Users/PC/Desktop/recipes.csv"
df = pd.read_csv(file_path)

# Vérifier que la colonne "total_time" existe
if "total_time" not in df.columns:
    raise ValueError("La colonne 'total_time' est introuvable dans le fichier CSV.")

# Fonction pour convertir le temps en minutes
def convert_to_minutes(time_str):
    if pd.isna(time_str):
        return np.nan
    time_str = time_str.lower().replace(" ", "")
    
    hours = re.search(r'(\d+)\s*(h|heure)', time_str)
    minutes = re.search(r'(\d+)\s*(m|min)', time_str)
    
    total_minutes = 0
    if hours:
        total_minutes += int(hours.group(1)) * 60
    if minutes:
        total_minutes += int(minutes.group(1))
    
    return total_minutes if total_minutes > 0 else np.nan

# Nettoyage des données
df = df[['recipe_name', 'rating', 'total_time']].dropna()
df = df.drop_duplicates(subset=['recipe_name'])
df['total_time'] = df['total_time'].apply(convert_to_minutes)

# Normalisation des notes
y_min, y_max = df['rating'].min(), df['rating'].max()
df['rating'] = (df['rating'] - y_min) / (y_max - y_min)

# Séparer les données
X = df['total_time'].values.reshape(-1, 1)
y = df['rating'].values

# Implémentation d'une régression avec un arbre de décision sans sklearn
def find_best_split(X, y):
    best_split = None
    best_variance = float('inf')
    unique_values = np.unique(X)
    
    for val in unique_values:
        left_mask = X.flatten() < val
        right_mask = X.flatten() >= val  
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            continue
        
        left_var = np.var(y[left_mask])
        right_var = np.var(y[right_mask])
        
        total_variance = left_var * np.sum(left_mask) + right_var * np.sum(right_mask)
        
        if total_variance < best_variance:
            best_variance = total_variance
            best_split = val
    
    return best_split

# Construction récursive de l'arbre
def build_tree(X, y, depth=0, max_depth=10):
    if depth >= max_depth or len(set(y)) == 1:
        return np.mean(y)
    
    split = find_best_split(X, y)
    if split is None:
        return np.mean(y)
    
    left_mask = X.flatten() < split
    right_mask = X.flatten() >= split
    
    return {
        'split': split,
        'left': build_tree(X[left_mask].reshape(-1, 1), y[left_mask], depth + 1, max_depth),
        'right': build_tree(X[right_mask].reshape(-1, 1), y[right_mask], depth + 1, max_depth)
    }

# Construire l'arbre de décision
tree = build_tree(X, y)

# Fonction de prédiction pour une seule valeur
def predict_from_tree(tree, total_time):
    total_time = convert_to_minutes(total_time)
    if np.isnan(total_time):
        return "Temps invalide"
    
    while isinstance(tree, dict):
        if total_time < tree['split']:
            tree = tree['left']
        else:
            tree = tree['right']
    
    return round(tree * (y_max - y_min) + y_min, 2)  # Dé-normalisation

# Fonction pour tester plusieurs valeurs en une seule exécution
def batch_predict(tree, time_list):
    results = {}
    for time in time_list:
        results[time] = predict_from_tree(tree, time)
    return results

# Exemple d'utilisation avec plusieurs valeurs
test_times = ["0m", "15m", "27h", "2h 17m", "90m"]
predictions = batch_predict(tree, test_times)

# Affichage des résultats
for time, rating in predictions.items():
    print(f"Temps : {time} → Prédiction de note : {rating}")
