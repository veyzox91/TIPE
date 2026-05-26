"""
╔══════════════════════════════════════════════════════════════╗
║  AMORTISSEUR DE MASSE ACCORDÉE (TMD) — TIPE                  ║
║  Méthode d'Euler explicite — Modèle 2 masses couplées        ║
╚══════════════════════════════════════════════════════════════╝

  PARTIE A — Réponse forcée
    • Simulation temporelle à la résonance (régime permanent)
    • Balayage fréquentiel → courbe de résonance

  PARTIE B — Décroissance libre
    • Impulsion initiale sur la plateforme (X₀, vitesse nulle)
    • Sortie : déplacement et accélération au cours du temps
    • L'accélération simulée est directement comparable aux mesures Phyphox

──────────────────────────────────────────────────────────────
MODÈLE PHYSIQUE
──────────────────────────────────────────────────────────────
Deux masses couplées via un ressort/amortisseur d'interface :
  - Plateforme (masse M) : ressort de structure K, amortisseur C
  - Masse TMD (masse m)  : ressort équivalent k, amortisseur c

Équations du mouvement (issues du PFD appliqué à chaque masse) :

  M·Ẍ = −K·X − C·Ẋ + k·(x−X) + c·(ẋ−Ẋ) + F(t)   [plateforme]
  m·ẍ = −k·(x−X) − c·(ẋ−Ẋ)                         [masse TMD]

Le terme k·(x−X) est la force de liaison (rappel du pendule linéarisé).
En décroissance libre : F(t) = 0, X(0) = X₀, tout le reste nul.
Phyphox mesure Ẍ(t) en m/s².
"""

import numpy as np
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════
# 1.  PARAMÈTRES PHYSIQUES DU SYSTÈME
# ══════════════════════════════════════════════════════════════

g = 9.81   # accélération de la pesanteur [m/s²]

# ── Plateforme principale ──────────────────────────────────────
M    = 0.500   # masse de la plateforme                     [kg]
K    = 40.0    # raideur des tiges de rappel                [N/m]
zeta = 0.02    # taux d'amortissement structurel (2 %)      [—]
               # Rappel : ζ = C / (2·M·ω₀), valeur typique pour bois

# Grandeurs déduites de M, K, zeta
omega0 = np.sqrt(K / M)            # pulsation propre ω₀ = √(K/M)  [rad/s]
f0     = omega0 / (2 * np.pi)      # fréquence propre f₀            [Hz]
C      = 2 * zeta * omega0 * M     # coefficient d'amortissement C  [N·s/m]
                                   # (déduit de la définition de ζ)

# ── Pendule TMD ────────────────────────────────────────────────
# On utilise les formules d'accord optimal de Den Hartog (1928) :
#   ω_p = ω₀ / (1+μ)     (accord en fréquence)
#   ζ_p = √(3μ / 8(1+μ)³) (amortissement optimal)

mu       = 0.05                            # rapport de masse μ = m/M  [—]
m        = mu * M                          # masse du pendule           [kg]

omega_p  = omega0 / (1 + mu)               # pulsation propre du TMD   [rad/s]
f_p      = omega_p / (2 * np.pi)           # fréquence du TMD           [Hz]

# Raideur équivalente du pendule linéarisé :
# Pour un pendule de longueur L, la pulsation propre est ω_p = √(g/L),
# soit la raideur équivalente k = m·ω_p² (terme de rappel de la PFD tangentielle)
k        = m * omega_p**2                  # raideur équivalente        [N/m]

# Longueur à régler sur la maquette (relation ω_p² = g/L)
L_pend   = g / omega_p**2                  # longueur du pendule        [m]

# Amortissement optimal du TMD (formule de Den Hartog)
zeta_p   = np.sqrt(3 * mu / (8 * (1 + mu)**3))
c        = 2 * zeta_p * m * omega_p        # coefficient d'amortisseur  [N·s/m]

# ── Conditions initiales (Partie B — décroissance libre) ──────
X0 = 0.020   # déplacement initial de la plateforme : 2 cm  [m]
             # Modélise le fait de tirer et lâcher la plateforme.
             # Toutes les autres conditions initiales sont nulles
             # (plateforme au repos sauf ce déplacement ; pendule à l'équilibre).

# ── Excitation sinusoïdale (Partie A — réponse forcée) ────────
F0    = 1.0       # amplitude de la force extérieure          [N]
Omega = omega0    # on excite exactement à la résonance ω = ω₀ [rad/s]
                  # C'est là que l'effet du TMD est maximal.

# ── Affichage récapitulatif des paramètres ────────────────────
print("╔════════════════════════════════════════════ ╗")
print("║         PARAMÈTRES DU SYSTÈME               ║")
print("╠════════════════════════════════════════════ ╣")
print(f"║  f₀ = {f0:.3f} Hz   ω₀ = {omega0:.3f} rad/s║")
print(f"║  ζ  = {zeta:.3f}    C  = {C:.4f} N·s/m     ║")
print(f"║  μ  = {mu:.3f}    m  = {m*1000:.1f} g      ║")
print(f"║  ► L pendule = {L_pend*100:.2f} cm         ║")
print(f"║  f_TMD = {f_p:.3f} Hz  ζ_TMD = {zeta_p:.4f}║")
print("╚════════════════════════════════════════════ ╝\n")


# ══════════════════════════════════════════════════════════════
# 2.  INTÉGRATEUR — SCHÉMA D'EULER EXPLICITE
# ══════════════════════════════════════════════════════════════
#
# Principe du schéma d'Euler explicite (ordre 1) :
#   y(t + dt) ≈ y(t) + dt · y'(t)
#
# On transforme le système de 2 EDO d'ordre 2 en 4 EDO d'ordre 1
# (forme de Cauchy / variables d'état) :
#   état = (X, Ẋ, x, ẋ)    avec  Ẍ = f(X, Ẋ, x, ẋ, t)
#                                  ẍ = g(X, Ẋ, x, ẋ)
#
# À chaque pas i → i+1 :
#   Ẋ[i+1] = Ẋ[i] + dt · Ẍ[i]
#   X[i+1] = X[i] + dt · Ẋ[i]     ← on utilise la vitesse au pas i (Euler explicite)
#   (idem pour x, ẋ)
# ──────────────────────────────────────────────────────────────

def euler(mode, n_periodes=80, pts_par_periode=600,
          avec_tmd=True, F0_=0.0, Omega_=omega0,
          X0_=0.0, t_max_libre=None):
    """
    Intègre les équations du mouvement par le schéma d'Euler explicite.

    Paramètres
    ----------
    mode            : 'force'  → excitation sinusoïdale F₀·cos(Ω·t)
                      'libre'  → décroissance libre, F(t) = 0
    n_periodes      : nombre de périodes simulées (mode 'force' uniquement)
    pts_par_periode : nombre de pas d'Euler par période (résolution temporelle)
                      Règle empirique : au moins 100–200 pts/période pour Euler.
                      600 garantit une bonne précision.
    avec_tmd        : True → système 2 DDL couplé (avec le pendule)
                      False → oscillateur simple 1 DDL (plateforme seule)
    F0_             : amplitude de la force [N]
    Omega_          : pulsation d'excitation [rad/s]
    X0_             : déplacement initial de la plateforme [m]
    t_max_libre     : durée de simulation pour le mode 'libre' [s]

    Retourne
    --------
    T      : tableau des instants [s]
    X      : déplacement de la plateforme [m]
    x_tmd  : déplacement de la masse TMD [m]
    Xddot  : accélération de la plateforme [m/s²]
    """

    # ── Calcul du pas de temps dt et du nombre de pas N ──────
    if mode == 'force':
        # On choisit dt pour avoir exactement pts_par_periode points par période
        dt = (2 * np.pi / Omega_) / pts_par_periode
        N  = n_periodes * pts_par_periode
    else:
        # En mode libre, on s'appuie sur la période propre f₀ pour calibrer dt
        dt = 1.0 / (pts_par_periode * f0)
        N  = int(t_max_libre / dt)   # nombre total de pas entier

    # ── Allocation des tableaux (pré-allocation = bonne pratique) ──
    # Évite de redimensionner un tableau à chaque itération (coûteux).
    T      = np.zeros(N)   # temps
    X      = np.zeros(N)   # déplacement plateforme
    Xdot   = np.zeros(N)   # vitesse plateforme
    x_tmd  = np.zeros(N)   # déplacement masse TMD
    xdot   = np.zeros(N)   # vitesse masse TMD
    Xddot  = np.zeros(N)   # accélération plateforme (sortie pour Phyphox)

    # ── Conditions initiales ──────────────────────────────────
    X[0] = X0_
    # Toutes les autres grandeurs restent à 0 (déjà initialisées par np.zeros).

    # ── Boucle d'intégration ──────────────────────────────────
    for i in range(N - 1):   # de l'indice 0 à N-2 (on calcule l'état i+1)

        # Force extérieure au pas courant
        if mode == 'force':
            F = F0_ * np.cos(Omega_ * T[i])   # excitation harmonique
        else:
            F = 0.0                             # décroissance libre

        # ── Calcul des accélérations via le PFD ──────────────
        if avec_tmd:
            # Force d'interaction entre le TMD et la plateforme :
            # rappel élastique + frottement visqueux de l'amortisseur du TMD
            F_liaison = k * (x_tmd[i] - X[i]) + c * (xdot[i] - Xdot[i])

            # PFD sur la plateforme (M·Ẍ = somme des forces) :
            Xdd = (-K * X[i] - C * Xdot[i] + F_liaison + F) / M
            #       ↑ rappel     ↑ frottement  ↑ réaction TMD  ↑ excitation

            # PFD sur la masse TMD (m·ẍ = force de liaison en sens opposé) :
            xdd = (-k * (x_tmd[i] - X[i]) - c * (xdot[i] - Xdot[i])) / m
            #  Signe opposé à F_liaison : 3e loi de Newton (action-réaction).
        else:
            # Sans TMD : oscillateur simple amorti 1 DDL
            Xdd = (-K * X[i] - C * Xdot[i] + F) / M
            xdd = 0.0   # la masse TMD n'existe pas, on garde 0 par convention

        # Stockage de l'accélération au pas courant (utile pour la comparaison Phyphox)
        Xddot[i] = Xdd

        # ── Mise à jour par le schéma d'Euler explicite ──────
        # On calcule d'abord la vitesse (car elle dépend de l'accélération),
        # puis le déplacement (car il dépend de la vitesse au pas précédent).
        Xdot[i+1]  = Xdot[i]  + dt * Xdd        # vitesse plateforme
        X[i+1]     = X[i]     + dt * Xdot[i]    # déplacement plateforme
        xdot[i+1]  = xdot[i]  + dt * xdd        # vitesse TMD
        x_tmd[i+1] = x_tmd[i] + dt * xdot[i]   # déplacement TMD
        T[i+1]     = T[i]     + dt              # avancement du temps

    # Remplissage du dernier point d'accélération (non calculé dans la boucle)
    Xddot[-1] = Xddot[-2]

    return T, X, x_tmd, Xddot


# ══════════════════════════════════════════════════════════════
# 3.  PARTIE B — DÉCROISSANCE LIBRE
# ══════════════════════════════════════════════════════════════
# On donne une impulsion initiale X₀ à la plateforme (ẋ=0, x_tmd=0)
# et on observe comment l'oscillation se dissipe, avec et sans TMD.
# La durée de simulation est choisie assez longue pour voir la décroissance
# jusqu'à une amplitude négligeable.

print("\nPARTIE B — Décroissance libre...")

t_sim = 20.0   # durée de simulation                              [s]

# Simulation sans TMD (oscillateur 1 DDL)
T_ls, X_ls, _,    Add_ls = euler('libre', avec_tmd=False, X0_=X0, t_max_libre=t_sim)

# Simulation avec TMD (système 2 DDL couplé)
T_la, X_la, xa_l, Add_la = euler('libre', avec_tmd=True,  X0_=X0, t_max_libre=t_sim)

print("  Terminé ✓")


# ══════════════════════════════════════════════════════════════
# 4.  DONNÉES PHYPHOX — À REMPLIR AVEC TES MESURES RÉELLES
# ══════════════════════════════════════════════════════════════
#
# Exporte tes données depuis Phyphox (bouton "Share" → CSV).
# Tu obtiens un fichier avec deux colonnes : temps [s] et accélération [m/s²].
#
# Option 1 — coller les valeurs directement dans le script :
#   t_phyphox     = np.array([0.0, 0.01, 0.02, ...])
#   accel_phyphox = np.array([0.1, 0.3, ...])
#
# Option 2 — charger un fichier CSV exporté par Phyphox :
#   import pandas as pd
#   df = pd.read_csv("ma_mesure.csv", sep=";")
#   t_phyphox     = df["Zeit (s)"].values
#   accel_phyphox = df["Beschleunigung (m/s²)"].values
#
# ─────────────────────────────────────────────────────────────
# En attendant tes vraies mesures, on génère des données fictives
# = simulation + bruit gaussien (σ = 0.08 m/s²), pour montrer
# à quoi ressemblera la superposition théorie/expérience.
# Remplace ces lignes par tes données Phyphox une fois disponibles.

np.random.seed(0)   # graine fixe → résultats reproductibles

# Données fictives "sans TMD" : on sous-échantillonne la simulation (1 pt sur 5)
# et on ajoute un bruit gaussien centré, d'écart-type 0.08 m/s²
t_phyphox_sans     = T_ls[::5]
accel_phyphox_sans = Add_ls[::5] + 0.08 * np.random.randn(len(T_ls[::5]))

# Données fictives "avec TMD"
t_phyphox_avec     = T_la[::5]
accel_phyphox_avec = Add_la[::5] + 0.08 * np.random.randn(len(T_la[::5]))


# ══════════════════════════════════════════════════════════════
# 5.  PARTIE A — RÉPONSE FORCÉE
# ══════════════════════════════════════════════════════════════

# ── 5a. Simulation temporelle à la résonance ─────────────────
# On excite à Ω = ω₀ (pire cas) et on attend le régime permanent
# (la partie transitoire est ignorée en ne prenant que la 2e moitié).
print("PARTIE A — Réponse forcée à la résonance...")

T_fs, X_fs, _, _     = euler('force', avec_tmd=False, F0_=F0, Omega_=Omega)
T_fa, X_fa, x_fa, _ = euler('force', avec_tmd=True,  F0_=F0, Omega_=Omega)

# On ignore la première moitié (transitoire) et on mesure l'amplitude en régime permanent
mi = len(T_fs) // 2              # indice du milieu (début du régime permanent)
amp_sans = np.max(np.abs(X_fs[mi:]))   # amplitude crête sans TMD
amp_avec = np.max(np.abs(X_fa[mi:]))   # amplitude crête avec TMD
reduct   = (1 - amp_avec / amp_sans) * 100   # réduction en %

print(f"  Sans TMD : {amp_sans*1e3:.1f} mm   Avec TMD : {amp_avec*1e3:.1f} mm   → −{reduct:.0f}%")

# ── 5b. Balayage fréquentiel → amplitude ET phase ────────────
# Pour chaque fréquence d'excitation Ω, on simule jusqu'au régime permanent
# puis on extrait l'amplitude ET la phase de X(t) par projection de Fourier
# sur les n_periodes_perm dernières périodes (régime établi).
#
# Principe de la projection (en régime permanent, X est sinusoïdal à Ω) :
#   X(t) = a·cos(Ωt) + b·sin(Ωt) = |X|·cos(Ωt − φ)
# avec   a = (2/T)∫X·cos(Ωt)dt ,  b = (2/T)∫X·sin(Ωt)dt
#        |X| = √(a² + b²) ,        φ = atan2(b, a)
#
# L'intégration sur un nombre ENTIER de périodes est essentielle pour la précision.

print("  Balayage fréquentiel (amplitude + phase)...")

n_freq          = 120                                          # nombre de fréquences sondées
omegas          = np.linspace(0.4 * omega0, 1.7 * omega0, n_freq)   # plage [0.4 ω₀ ; 1.7 ω₀]
pts_per_period  = 400                                          # pas Euler par période
n_periodes_tot  = 45                                           # durée totale (transitoire + permanent)
n_periodes_perm = 20                                           # périodes utilisées pour la projection

A_sans   = np.zeros(n_freq)    # amplitudes sans TMD
A_avec   = np.zeros(n_freq)    # amplitudes avec TMD
phi_sans = np.zeros(n_freq)    # phase de X(t) par rapport à F(t), sans TMD
phi_avec = np.zeros(n_freq)    # phase de X(t) par rapport à F(t), avec TMD

for i, om in enumerate(omegas):
    dt = (2 * np.pi / om) / pts_per_period   # pas de temps adapté à chaque fréquence
    N  = n_periodes_tot * pts_per_period
    T_ = np.zeros(N)

    # Tableaux d'état pour les deux simulations (1 DDL et 2 DDL) en parallèle
    Xs  = np.zeros(N); Xds  = np.zeros(N)   # plateforme sans TMD
    Xa  = np.zeros(N); Xda  = np.zeros(N)   # plateforme avec TMD
    xa  = np.zeros(N); xda  = np.zeros(N)   # masse TMD

    for j in range(N - 1):
        F = F0 * np.cos(om * T_[j])   # force harmonique au pas j

        # ── Sans TMD (1 DDL) ─────────────────────────────────
        Xdds     = (-K * Xs[j] - C * Xds[j] + F) / M
        Xds[j+1] = Xds[j] + dt * Xdds
        Xs[j+1]  = Xs[j]  + dt * Xds[j]

        # ── Avec TMD (2 DDL couplés) ─────────────────────────
        Fl       = k * (xa[j] - Xa[j]) + c * (xda[j] - Xda[j])   # force d'interface
        Xdda     = (-K * Xa[j] - C * Xda[j] + Fl + F) / M
        xdda     = (-k * (xa[j] - Xa[j]) - c * (xda[j] - Xda[j])) / m
        Xda[j+1] = Xda[j] + dt * Xdda
        Xa[j+1]  = Xa[j]  + dt * Xda[j]
        xda[j+1] = xda[j] + dt * xdda
        xa[j+1]  = xa[j]  + dt * xda[j]
        T_[j+1]  = T_[j]  + dt

    # ── Extraction amplitude + phase par projection de Fourier ──
    # On sélectionne les n_periodes_perm dernières périodes (régime établi)
    i_start = (n_periodes_tot - n_periodes_perm) * pts_per_period
    t_perm  = T_[i_start:]
    cos_ref = np.cos(om * t_perm)
    sin_ref = np.sin(om * t_perm)

    # Cas sans TMD
    a_s = 2 * np.mean(Xs[i_start:] * cos_ref)
    b_s = 2 * np.mean(Xs[i_start:] * sin_ref)
    A_sans[i]   = np.sqrt(a_s**2 + b_s**2)
    phi_sans[i] = np.arctan2(b_s, a_s)

    # Cas avec TMD
    a_a = 2 * np.mean(Xa[i_start:] * cos_ref)
    b_a = 2 * np.mean(Xa[i_start:] * sin_ref)
    A_avec[i]   = np.sqrt(a_a**2 + b_a**2)
    phi_avec[i] = np.arctan2(b_a, a_a)

print("  Terminé ✓")


# ══════════════════════════════════════════════════════════════
# 5c.  FONCTION DE TRANSFERT ANALYTIQUE  H(jΩ) = X̃ / F̃
# ══════════════════════════════════════════════════════════════
#
# Mise en équation harmonique : on suppose X(t) = Re(X̃·e^{jΩt}),
# F(t) = Re(F̃·e^{jΩt}) ; les dérivées deviennent :
#      Ẋ → jΩ·X̃        Ẍ → −Ω²·X̃
#
# Les EDO de mouvement (cf. modèle physique) deviennent algébriques :
#   (−MΩ² + K + jΩC)·X̃ = Z_s·(x̃ − X̃) + F̃          ...(plateforme)
#   −mΩ²·x̃              = −Z_s·(x̃ − X̃)              ...(TMD)
#
# avec Z_s = k + jΩc , l'impédance complexe du couplage TMD.
#
# (2) ⟹  x̃ = Z_s·X̃ / (Z_s − mΩ²)
#
# En injectant dans (1) et en isolant X̃/F̃ :
#
#                              1
#   H(jΩ) = ─────────────────────────────────────────────────
#           K − MΩ² + jΩC − mΩ²·Z_s / (Z_s − mΩ²)
#
# Sans TMD (m = 0) : H₀(jΩ) = 1 / (K − MΩ² + jΩC)   ← oscillateur 1 DDL classique
# ──────────────────────────────────────────────────────────────

Z_s        = k + 1j * omegas * c
H_th_avec  = 1 / (K - M*omegas**2 + 1j*omegas*C
                   - m*omegas**2 * Z_s / (Z_s - m*omegas**2))
H_th_sans  = 1 / (K - M*omegas**2 + 1j*omegas*C)

# Gain en dB, normalisé par la déflexion statique X_stat = F₀/K
# (à Ω → 0, K·H → 1, donc gain → 0 dB : référence naturelle)
G_th_sans_dB = 20 * np.log10(np.abs(K * H_th_sans))
G_th_avec_dB = 20 * np.log10(np.abs(K * H_th_avec))

# Gain reconstruit à partir des amplitudes simulées
G_sim_sans_dB = 20 * np.log10(K * A_sans / F0)
G_sim_avec_dB = 20 * np.log10(K * A_avec / F0)

# Phase de H(jΩ) : arg(H) = −φ (car F̃ = F₀ réel positif, X̃ = |X|·e^{−jφ})
# np.unwrap retire les sauts de ±2π pour obtenir une courbe continue
phase_th_sans_deg  = np.degrees(np.unwrap(np.angle(H_th_sans)))
phase_th_avec_deg  = np.degrees(np.unwrap(np.angle(H_th_avec)))
phase_sim_sans_deg = -np.degrees(np.unwrap(phi_sans))
phase_sim_avec_deg = -np.degrees(np.unwrap(phi_avec))


# ══════════════════════════════════════════════════════════════
# 6.  TRACÉS
# ══════════════════════════════════════════════════════════════

# ── Palette de couleurs et style global ──────────────────────
ROUGE = "#c0392b"
BLEU  = "#1a4fa0"
VERT  = "#1a7a45"

plt.rcParams.update({
    "font.family"        : "serif",    # police avec empattements, plus lisible en impression
    "font.size"          : 11,
    "axes.grid"          : True,
    "grid.alpha"         : 0.3,
    "grid.linestyle"     : "--",
    "axes.spines.top"    : False,      # suppression des bordures inutiles (style épuré)
    "axes.spines.right"  : False,
})

# Figure principale : 3 lignes × 2 colonnes
# gs[0,0] → A1  |  gs[0,1] → A2
# gs[1,:] → B1 (ligne entière)
# gs[2,:] → B2 (ligne entière)
fig = plt.figure(figsize=(14, 16))
fig.suptitle(
    "Amortisseur de Masse Accordée (TMD) — Simulation complète\n"
    f"M = {M*1000:.0f} g,  m = {m*1000:.0f} g,  "
    f"f₀ = {f0:.2f} Hz,  L = {L_pend*100:.1f} cm",
    fontsize=13, fontweight="bold"
)

gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.35)

# ── A1 : Réponse temporelle forcée (régime permanent) ─────────
# On ne trace que la seconde moitié (mi:) pour éviter le transitoire.
ax = fig.add_subplot(gs[0, 0])
ax.plot(T_fs[mi:], X_fs[mi:] * 1e3, ROUGE, lw=1.3,
        label=f"Sans TMD  ({amp_sans*1e3:.1f} mm)")
ax.plot(T_fa[mi:], X_fa[mi:] * 1e3, BLEU,  lw=1.3,
        label=f"Avec TMD  ({amp_avec*1e3:.1f} mm,  −{reduct:.0f}%)")
ax.set(xlabel="Temps [s]", ylabel="Déplacement X [mm]",
       title="A1 — Réponse forcée à la résonance\n(régime permanent)")
ax.legend(fontsize=9)

# ── A2 : Courbe de résonance |X(Ω)| ──────────────────────────
# L'axe des abscisses est normalisé par ω₀ : Ω/ω₀ = 1 correspond à la résonance.
# Avec TMD, on observe la scission du pic en deux pics secondaires (effet "anti-résonance").
ax = fig.add_subplot(gs[0, 1])
fn = omegas / omega0   # fréquence réduite (adimensionnée)
ax.plot(fn, A_sans * 1e3, ROUGE, lw=2, label="Sans TMD")
ax.plot(fn, A_avec * 1e3, BLEU,  lw=2, label="Avec TMD")
ax.axvline(1.0, color="gray", lw=0.9, ls="--", alpha=0.7)   # repère ω = ω₀
ax.set(xlabel="Fréquence normalisée  Ω/ω₀", ylabel="Amplitude [mm]",
       title="A2 — Courbe de résonance")
ax.legend(fontsize=9)
ax.set_ylim(0, None)

# ── B1 : Décroissance libre — déplacement ─────────────────────
# Montre directement que le TMD accélère la dissipation de l'énergie
# de la plateforme : la courbe bleue retombe beaucoup plus vite.
ax = fig.add_subplot(gs[1, :])   # gs[1,:] → occupe toute la 2e ligne
ax.plot(T_ls, X_ls * 1e3, ROUGE, lw=1.2, alpha=0.9, label="Sans TMD — simulation")
ax.plot(T_la, X_la * 1e3, BLEU,  lw=1.2, alpha=0.9, label="Avec TMD — simulation")
ax.set(xlabel="Temps [s]", ylabel="Déplacement X [mm]",
       title=f"B1 — Décroissance libre (impulsion X₀ = {X0*100:.0f} cm)\n"
             "     Avec TMD : la décroissance est beaucoup plus rapide")
ax.legend(fontsize=9)

# Annotation : ligne horizontale au niveau de la demi-amplitude sans TMD
# La constante de temps τ ≈ M/C donne l'ordre de grandeur de la décroissance exponentielle.
tau_sans = M / C   # constante de temps approx. de l'oscillateur amorti sans TMD  [s]
ax.axhline(X0 * 1e3 / 2, color=ROUGE, lw=0.7, ls=":", alpha=0.5)
ax.text(tau_sans * 0.7, X0 * 1e3 / 2 + 0.5,
        "A₀/2 sans TMD", color=ROUGE, fontsize=8)

# ── B2 : Décroissance libre — ACCÉLÉRATION ────────────────────
# C'est la grandeur mesurée par Phyphox.
# Superposer simulation et points expérimentaux permet de valider le modèle.
ax = fig.add_subplot(gs[2, :])
ax.plot(T_ls, Add_ls, ROUGE, lw=1.2, alpha=0.85, label="Sans TMD — simulation")
ax.plot(T_la, Add_la, BLEU,  lw=1.2, alpha=0.85, label="Avec TMD — simulation")

# Points Phyphox (fictifs pour l'instant → à remplacer par tes mesures réelles)
# On sous-échantillonne encore (1 pt sur 3) pour ne pas surcharger le graphe.
ax.scatter(t_phyphox_sans[::3], accel_phyphox_sans[::3],
           color=ROUGE, s=8, alpha=0.4,
           label="Sans TMD — Phyphox (fictif, à remplacer)")
ax.scatter(t_phyphox_avec[::3], accel_phyphox_avec[::3],
           color=BLEU,  s=8, alpha=0.4,
           label="Avec TMD — Phyphox (fictif, à remplacer)")

ax.set(xlabel="Temps [s]", ylabel="Accélération [m/s²]",
       title="B2 — Décroissance libre : ACCÉLÉRATION\n"
             "     ← directement comparable à la mesure Phyphox →")
ax.legend(fontsize=9, ncol=2)

# Axe droit en unités g (1 g = 9.81 m/s²) pour faciliter la lecture physique
ax2 = ax.twinx()
ax2.set_ylim(np.array(ax.get_ylim()) / g)   # conversion linéaire des limites
ax2.set_ylabel("Accélération [g]")
ax2.spines[["top"]].set_visible(False)

plt.savefig("tmd_complet.png", dpi=150, bbox_inches="tight", facecolor="white")
print("\nFigure sauvegardée → tmd_complet.png")


# ══════════════════════════════════════════════════════════════
# 7.  RÉSUMÉ CONSOLE
# ══════════════════════════════════════════════════════════════
print("\n╔════════════════════════════════════════════╗")
print("║            RÉSULTATS                       ║")
print("╠════════════════════════════════════════════╣")
print(f"║  Réduction amplitude forcée  : −{reduct:.0f} %       ║")

# Temps pour que l'amplitude de la plateforme tombe sous 10 % de X₀
# (seuil arbitraire mais parlant : −90 % de l'oscillation initiale)
seuil = 0.10 * X0

# next(...) renvoie le premier indice i tel que |X_ls[i]| < seuil (après les 10 premiers)
# Si l'amplitude ne passe jamais sous le seuil, next() renvoie None.
idx_s = next((i for i, v in enumerate(np.abs(X_ls)) if v < seuil and i > 10), None)
idx_a = next((i for i, v in enumerate(np.abs(X_la)) if v < seuil and i > 10), None)

if idx_s:
    print(f"║  Temps décroissance −90% sans TMD : {T_ls[idx_s]:.1f} s  ║")
if idx_a:
    print(f"║  Temps décroissance −90% avec TMD : {T_la[idx_a]:.1f} s  ║")

print(f"║  ► Longueur pendule à régler  : {L_pend*100:.2f} cm   ║")
print("╚════════════════════════════════════════════╝")


# ══════════════════════════════════════════════════════════════
# 8.  FIGURE BODE : Gain (dB) + Phase (degrés)
# ══════════════════════════════════════════════════════════════
# Superposition simulation (points) ↔ théorie analytique (trait plein).
# C'est la signature visuelle classique du TMD :
#   • un creux profond (anti-résonance) à Ω ≈ ω_p
#   • encadré par deux pics secondaires (modes couplés du système 2 DDL)
#   • signature de phase complexe autour de ω_p (plateau/inversion locale),
#     malgré une asymptote commune à −180° pour Ω → ∞
# ──────────────────────────────────────────────────────────────

fig_bode, (ax_g, ax_p) = plt.subplots(
    2, 1, figsize=(11, 8.5), sharex=True,
    gridspec_kw={"hspace": 0.12, "height_ratios": [1.2, 1]}
)
fig_bode.suptitle(
    "Diagramme de Bode — fonction de transfert  H(jΩ) = X̃ / F̃\n"
    "Validation croisée : points = simulation Euler   |   trait plein = expression analytique",
    fontsize=12, fontweight="bold"
)

fn = omegas / omega0   # fréquence réduite Ω/ω₀

# ── Gain (en haut) ───────────────────────────────────────────
ax_g.plot(fn, G_th_sans_dB, ROUGE, lw=2, label="Sans TMD — théorie")
ax_g.plot(fn, G_th_avec_dB, BLEU,  lw=2, label="Avec TMD — théorie")
ax_g.scatter(fn[::4], G_sim_sans_dB[::4], color=ROUGE, s=22, zorder=5,
             edgecolor="white", linewidth=0.6, label="Sans TMD — simulation Euler")
ax_g.scatter(fn[::4], G_sim_avec_dB[::4], color=BLEU,  s=22, zorder=5,
             edgecolor="white", linewidth=0.6, label="Avec TMD — simulation Euler")
ax_g.axvline(1.0, color="gray", lw=0.9, ls="--", alpha=0.6)
ax_g.axhline(0.0, color="gray", lw=0.7, ls=":", alpha=0.5)
# Repère de l'anti-résonance (Ω = ω_p)
ax_g.axvline(omega_p / omega0, color=VERT, lw=0.9, ls="--", alpha=0.5)
ax_g.text(omega_p/omega0 - 0.02, ax_g.get_ylim()[1] if False else 25,
          "Ω = ω_p\n(anti-résonance)", color=VERT, fontsize=8, ha="right")
ax_g.text(1.02, 25, "Ω = ω₀", color="gray", fontsize=8)
ax_g.set_ylabel("Gain  20·log₁₀(K·|X|/F₀)  [dB]")
ax_g.set_title("Anti-résonance : creux profond à Ω ≈ ω_p, encadré par deux pics secondaires",
               fontsize=10, color="#444")
ax_g.legend(fontsize=9, loc="lower left")

# ── Phase (en bas) ───────────────────────────────────────────
ax_p.plot(fn, phase_th_sans_deg, ROUGE, lw=2, label="Sans TMD — théorie")
ax_p.plot(fn, phase_th_avec_deg, BLEU,  lw=2, label="Avec TMD — théorie")
ax_p.scatter(fn[::4], phase_sim_sans_deg[::4], color=ROUGE, s=22, zorder=5,
             edgecolor="white", linewidth=0.6)
ax_p.scatter(fn[::4], phase_sim_avec_deg[::4], color=BLEU,  s=22, zorder=5,
             edgecolor="white", linewidth=0.6)
ax_p.axvline(1.0, color="gray", lw=0.9, ls="--", alpha=0.6)
ax_p.axvline(omega_p / omega0, color=VERT, lw=0.9, ls="--", alpha=0.5)
# Repères horizontaux aux multiples de 90°
for y in (-90, -180, -270):
    ax_p.axhline(y, color="gray", lw=0.6, ls=":", alpha=0.4)
ax_p.set_xlabel("Fréquence normalisée  Ω / ω₀")
ax_p.set_ylabel("Phase  arg(H)  [°]")
ax_p.set_yticks([0, -45, -90, -135, -180, -225])
ax_p.set_ylim(-225, 30)
ax_p.set_title("Phase plate à basse fréquence ; saut rapide à la résonance ; "
               "signature ondulée du TMD autour de Ω = ω_p",
               fontsize=10, color="#444")

plt.savefig("tmd_bode.png", dpi=150, bbox_inches="tight", facecolor="white")
print("\nFigure Bode sauvegardée → tmd_bode.png")

plt.show()
