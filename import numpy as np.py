import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques
tmax = 10           # Durée de la simulation (s)
n = 2000            # Nombre de points
dt = tmax/n         # Pas de temps (s)  


# Propriétés de la structure (Bâtiment)
Ms = 5.0            # Masse (kg)
Ks = 2000.0         # Raideur (N/m)
Cs = 1.5            # Amortissement structurel (N.s/m)

# Propriétés des amortisseurs (on prend 10% de la masse de la structure)
ma = 0.5            # Masse de l'amortisseur (kg)
# On accorde l'amortisseur sur la fréquence de la structure : omega = sqrt(K/M)
ka = ma * (Ks / Ms) # Raideur (ou équivalent pendulaire)
ca_lin = 5.0        # Coefficient d'amortissement linéaire (TMD)
xi_liq = 0.8        # Coeff de perte de charge (Liquide - non linéaire)

# --- 2. INITIALISATION DES VECTEURS ---
t = np.linspace(0, t_final, n_steps)

# Fonctions d'initialisation des états [déplacement, vitesse]
def init_state():
    return np.zeros(n_steps), np.zeros(n_steps)

# États pour chaque scénario (s = structure, a = amortisseur)
xs_vide, vs_vide = init_state()
xs_tmd,  vs_tmd,  xa_tmd,  va_tmd  = init_state(), init_state(), init_state(), init_state()
xs_liq,  vs_liq,  xa_liq,  va_liq  = init_state(), init_state(), init_state(), init_state()

# Condition initiale : on écarte la structure de 10 cm
xs_vide[0] = xs_tmd[0] = xs_liq[0] = 0.1

# --- 3. RÉSOLUTION PAR MÉTHODE D'EULER ---
for i in range(n_steps - 1):
    
    # --- CAS 1 : STRUCTURE SEULE ---
    as_vide = (-Ks * xs_vide[i] - Cs * vs_vide[i]) / Ms
    vs_vide[i+1] = vs_vide[i] + as_vide * dt
    xs_vide[i+1] = xs_vide[i] + vs_vide[i] * dt

    # --- CAS 2 : MASSE-RESSORT / PENDULE (TMD) ---
    # Force de liaison (ressort + amortisseur linéaire)
    F_tmd = ka * (xa_tmd[i] - xs_tmd[i]) + ca_lin * (va_tmd[i] - vs_tmd[i])
    
    as_tmd = (-Ks * xs_tmd[i] - Cs * vs_tmd[i] + F_tmd) / Ms
    aa_tmd = (-F_tmd) / ma
    
    vs_tmd[i+1] = vs_tmd[i] + as_tmd * dt
    xs_tmd[i+1] = xs_tmd[i] + vs_tmd[i] * dt
    va_tmd[i+1] = va_tmd[i] + aa_tmd * dt
    xa_tmd[i+1] = xa_tmd[i] + va_tmd[i] * dt

    # --- CAS 3 : AMORTISSEUR LIQUIDE (TLCD) ---
    # Force de liaison avec terme quadratique : F = k*x + c*v*|v|
    v_rel = va_liq[i] - vs_liq[i]
    F_liq = ka * (xa_liq[i] - xs_liq[i]) + xi_liq * v_rel * abs(v_rel)
    
    as_liq = (-Ks * xs_liq[i] - Cs * vs_liq[i] + F_liq) / Ms
    aa_liq = (-F_liq) / ma
    
    vs_liq[i+1] = vs_liq[i] + as_liq * dt
    xs_liq[i+1] = xs_liq[i] + vs_liq[i] * dt
    va_liq[i+1] = va_liq[i] + aa_liq * dt
    xa_liq[i+1] = xa_liq[i] + va_liq[i] * dt

# --- 4. AFFICHAGE DES RÉSULTATS ---
plt.figure(figsize=(12, 7))
plt.plot(t, xs_vide, 'grey', alpha=0.5, label='Structure Seule (Référence)')
plt.plot(t, xs_tmd, 'blue', label='Masse-Ressort / Pendule (Linéaire)')
plt.plot(t, xs_liq, 'green', label='Liquide (TLCD - Quadratique)')

plt.title("Comparaison de l'efficacité des amortisseurs (Méthode d'Euler)")
plt.xlabel("Temps (s)")
plt.ylabel("Déplacement de la structure (m)")
plt.axhline(0, color='black', lw=1)
plt.legend()
plt.grid(True)
plt.show()