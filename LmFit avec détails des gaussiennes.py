import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.signal import find_peaks, peak_widths
import seaborn as sns

# Données fournies
x_data = [
    [0, 0.56, 1.56, 2.56, 3.56, 4.56, 5.56, 6.56, 7.56, 8.56, 9.56,
     10.56, 10.76, 10.96, 11.16, 11.36, 11.56, 11.76, 11.96, 12.16, 
     12.36, 12.56, 12.76, 12.96, 13.96, 14.96, 15.96, 16.96, 17.96, 
     19.96, 21.96, 25.6],
    [0, 1, 2.57, 3.07, 3.57, 4.07, 4.57, 5.07, 5.57, 6.07, 6.57, 
     7.07, 7.57, 7.77, 7.97, 8.17, 8.37, 8.57, 9.07, 9.57, 10.07, 
     10.57, 11.07, 12.07, 13.07, 14.07, 15.07, 17.07, 19.07, 22, 25]
]

y_data = [
    [222, 241, 277, 314, 354, 381, 378, 348, 357, 320, 319, 409, 447,
     469, 484, 496, 502, 503, 498, 488, 473, 456, 437, 416, 322, 253,
     203, 169, 142, 104, 85, 52],
    [677, 687, 691, 691, 693, 682, 680, 702, 715, 711, 728, 741, 711,
     714, 721, 727, 722, 715, 711, 697, 685, 676, 672, 655, 618, 569, 
     519, 434, 371, 306, 256]
]

# Incertitudes
error_p = [
    [221, 240, 276, 313, 353, 380, 377, 347, 336, 316, 316, 408, 446, 468, 483, 
     495, 501, 500, 497, 487, 473, 455, 436, 415, 320, 252, 202, 168, 140, 103, 
     84, 50],  # min incertitude
    [223, 242, 278, 315, 355, 382, 380, 349, 370, 324, 322, 410, 448, 470, 486, 
     498, 504, 504, 499, 488, 474, 457, 438, 417, 323, 254, 204, 170, 143, 105, 
     86, 53]   # max incertitude
]

error_e = [
    [676, 686, 691, 690, 692, 681, 679, 700, 713, 709, 727, 740, 709,
     713, 717, 720, 715, 714, 709, 696, 684, 675, 671, 654, 617, 568,
     518, 432, 370, 305, 254],  # min incertitude
    [678, 688, 692, 692, 694, 684, 681, 704, 716, 713, 729, 743, 712,
     715, 728, 733, 730, 716, 713, 698, 686, 677, 673, 656, 620, 570,
     520, 436, 372, 307, 257]   # max incertitude
]

results_summary = []

# Définir la fonction gaussienne
def gaussian(x, mu, sigma, amplitude):
    """Fonction gaussienne pour ajustement."""
    return amplitude * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Ajustement pour chaque ensemble de données
for i in range(len(x_data)):
    # Trouver les pics dans les données y
    position = find_peaks(y_data[i])[0]  # Indices des pics
    if len(position) == 0:
        print(f"Aucun pic trouvé pour l'ensemble de données {i + 1}.")
        continue  # Passer à l'ensemble de données suivant si aucun pic n'est trouvé

    # Initialisation des paramètres pour le modèle gaussien
    mu = [x_data[i][pos] for pos in position]  # Valeurs de x aux pics
    amplitude = [y_data[i][pos] for pos in position]  # Amplitude à ces pics
    sigma = peak_widths(y_data[i], position, rel_height=0.99)[0]  # Largeur des pics
    print("Moyennes (mu):", mu, "Amplitudes:", amplitude, "Largeurs (sigma):", sigma)

    # Création du modèle gaussien cumulatif
    model = None
    for j in range(len(mu)):
        prefix = f'peak_{j + 1}_'
        mod = Model(gaussian, prefix=prefix)
        model = mod if model is None else model + mod  # Combiner les modèles gaussiens

    # Créer des paramètres initiaux pour le modèle
    params = model.make_params()
    
    # Assignation des valeurs initiales
    for j in range(len(mu)):
        params.add(f'peak_{j + 1}_mu', value=mu[j])  # Valeur de mu pour le pic j
        params.add(f'peak_{j + 1}_sigma', value=sigma[j])  # Valeur de sigma pour le pic j
        params.add(f'peak_{j + 1}_amplitude', value=amplitude[j])  # Valeur d'amplitude pour le pic j

    # Calculer les incertitudes à partir des valeurs d'erreur
    y_min = np.array(error_p[0]) if i == 0 else np.array(error_e[0])  # Choisir les incertitudes appropriées
    y_max = np.array(error_p[1]) if i == 0 else np.array(error_e[1])
    
    y_errors_sup = y_max - y_data[i]  # Erreur positive
    y_errors_inf = y_data[i] - y_min  # Erreur négative

    # S'assurer que les erreurs ne sont pas négatives
    y_errors_inf = np.maximum(y_errors_inf, 0)  # Forcer à 0 si négatif
    y_errors_sup = np.maximum(y_errors_sup, 0)  # Forcer à 0 si négatif

    # Ajuster le modèle sur les données
    try:
        result = model.fit(y_data[i], params, x=x_data[i])
        print(result.fit_report())

        # Stocker les résultats
        for j in range(len(mu)):
            results_summary.append({
                'Ensemble de données': i + 1,
                'Pic': j + 1,
                'mu': result.params[f'peak_{j + 1}_mu'].value,
                'sigma': result.params[f'peak_{j + 1}_sigma'].value,
                'amplitude': result.params[f'peak_{j + 1}_amplitude'].value,
            })
    
        # Créer un DataFrame pour Seaborn
        df = pd.DataFrame({
            'x': x_data[i],
            'y': y_data[i],
            'y_fit': result.best_fit,  # Ajustement du modèle
            'y_min': y_min,
            'y_max': y_max,
            'y_errors_inf': y_errors_inf,
            'y_errors_sup': y_errors_sup
        })
        
        # Plage de x pour le tracé
        x_fit = np.linspace(min(x_data[i]), max(x_data[i]), 1000) 
        
        # Appliquer un style de graphique avec Seaborn
        sns.set_style('whitegrid')  # Choisir le style
        sns.color_palette("dark:#5A9_r", as_cmap=True)  # Choisir une palette de couleurs

        # Tracer les données et l'ajustement
        plt.figure(figsize=(12, 8))  # Taille du graphique
        sns.scatterplot(data=df, x='x', y='y', label='Données d\'origine', color='blue')
        plt.errorbar(df['x'], df['y'], yerr=[df['y_errors_inf'], df['y_errors_sup']], 
                     fmt='none', ecolor='blue', capsize=5, label='Incertitudes')
        sns.lineplot(data=df, x='x', y='y_fit', label='Ajustement gaussien', color='red', linewidth=2)

        # Tracer chaque gaussienne
        for j in range(len(mu)):
            y_gaussian = gaussian(x_fit, result.params[f'peak_{j + 1}_mu'].value, 
                                  result.params[f'peak_{j + 1}_sigma'].value, 
                                  result.params[f'peak_{j + 1}_amplitude'].value)
            plt.plot(x_fit, y_gaussian, linestyle='--', label=f'Gaussienne {j + 1}', alpha=0.7)

        # Ajouter des titres et des labels
        plt.title(f"Ajustement Gaussien pour l'ensemble de données {i + 1}", fontsize=16)
        plt.xlabel("Distance", fontsize=14)
        plt.ylabel("Amplitude", fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()  # Afficher le graphique

    except Exception as e:
        print(f"Erreur lors de l'ajustement pour l'ensemble de données {i + 1}: {e}")

# Afficher le résumé des résultats
results_df = pd.DataFrame(results_summary)
print(results_df)
