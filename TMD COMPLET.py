"""
╔══════════════════════════════════════════════════════════════╗
║  AMORTISSEUR DE MASSE ACCORDÉE (TMD)                         ║
║  Méthode d'Euler — Modèle 2 masses couplées                  ║
╚══════════════════════════════════════════════════════════════╝

  PARTIE A — Réponse forcée
    • Simulation temporelle à la résonance
    • Balayage fréquentiel (courbe de résonance)

  PARTIE B — Décroissance libre
    • On donne une impulsion initiale à la plateforme
    • La simulation sort le déplacement et l'accélération au cours du temps
    • L'accélération simulée est directement comparable aus mesures réelles

MODÈLE PHYSIQUE
════════════════════════
Deux masses couplées : la plateforme (masse M) et le TMD (masse m)

  M·Ẍ = −K·X − C·Ẋ + k·(x−X) + c·(ẋ−Ẋ) + F(t)    [plateforme]
  m·ẍ = −k·(x−X) − c·(ẋ−Ẋ)                          [masse TMD]

En décroissance libre : F(t) = 0, condition initiale X(0) = X0, tout le reste nul.

Phyphox mesure Ẍ(t) en m/s²
"""

import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════
# PARAMÈTRES DU SYSTÈME
# ══════════════════════════════════════════════════════

g = 9.81   # [m/s²]

# ── Plateforme ─────────────────────────────────────
M    = 0.500   # [kg]   masse de la plateforme
K    = 40.0    # [N/m]  raideur des tiges
zeta = 0.02    # [—]    amortissement structurel

# Grandeurs dérivées
omega0 = np.sqrt(K / M)
f0     = omega0 / (2 * np.pi)
C      = 2 * zeta * omega0 * M

# ── Pendule TMD ────────────────────────────────────
mu       = 0.05
m        = mu * M
omega_p  = omega0 / (1 + mu)          # accord optimal
f_p      = omega_p / (2 * np.pi)
k        = m * omega_p**2             # raideur équivalente du pendule
L_pend   = g / omega_p**2            # longueur à régler
zeta_p   = np.sqrt(3*mu / (8*(1+mu)**3))
c        = 2 * zeta_p * m * omega_p

# ── Décroissance libre (Partie B) ───────────────────
X0 = 0.020   # [m]  déplacement initial de la plateforme = 2 cm (simule le fait de pousser la plateforme et de lâcher)

# ── Excitation (Partie B) ──────────────────────────
F0    = 1.0      # [N]      amplitude de la force
Omega = omega0   # [rad/s]  on excite à la résonance

print("╔════════════════════════════════════════════ ╗")
print("║         PARAMÈTRES DU SYSTÈME               ║")
print("╠════════════════════════════════════════════ ╣")
print(f"║  f₀ = {f0:.3f} Hz   ω₀ = {omega0:.3f} rad/s║")
print(f"║  ζ  = {zeta:.3f}    C  = {C:.4f} N·s/m     ║")
print(f"║  μ  = {mu:.3f}    m  = {m*1000:.1f} g      ║")
print(f"║  ► L pendule = {L_pend*100:.2f} cm         ║")
print(f"║  f_TMD = {f_p:.3f} Hz  ζ_TMD = {zeta_p:.4f}║")
print("╚════════════════════════════════════════════ ╝\n")


# ══════════════════════════════════════════════════════
# MÉTHODE D'EULER 
# ══════════════════════════════════════════════════════
"""
    Résout le système d'équations différentielles par la méthode d'Euler 

    mode = 'force'  : réponse à F0·cos(Ω·t), conditions initiales nulles
    mode = 'libre'  : décroissance libre,  conditions initiales = [X0, 0, 0, 0]

    Renvoie : T, X, x_tmd, Xddot
      T      = temps [s]
      X      = déplacement plateforme [m]
      x_tmd  = déplacement masse TMD [m]
      Xddot  = accélération plateforme [m/s²]
    """
def euler(mode, n_periodes=80, pts_par_periode=600,
          avec_tmd=True, F0_=0.0, Omega_=omega0,
          X0_=0.0, t_max_libre=None):
    
    if mode == 'force':
        dt    = (2 * np.pi / Omega_) / pts_par_periode
        N     = n_periodes * pts_par_periode
    else:
        dt    = 1.0 / (pts_par_periode * f0)   # pas fin même en libre
        N     = int(t_max_libre / dt)

    T      = np.zeros(N)
    X      = np.zeros(N)
    Xdot   = np.zeros(N)
    x_tmd  = np.zeros(N)
    xdot   = np.zeros(N)
    Xddot  = np.zeros(N)

    # Conditions initiales
    X[0] = X0_

    for i in range(N - 1):
        # Force extérieure
        if mode == 'force':
            F = F0_ * np.cos(Omega_ * T[i])
        else:
            F = 0.0

        # Calcul des accélérations (2e loi de Newton)
        if avec_tmd:
            F_liaison = k * (x_tmd[i] - X[i]) + c * (xdot[i] - Xdot[i])
            Xdd = (-K*X[i] - C*Xdot[i] + F_liaison + F) / M
            xdd = (-k*(x_tmd[i]-X[i]) - c*(xdot[i]-Xdot[i])) / m
        else:
            Xdd = (-K*X[i] - C*Xdot[i] + F) / M
            xdd = 0.0

        # Stockage de l'accélération
        Xddot[i] = Xdd

        # Schéma d'Euler
        Xdot[i+1]  = Xdot[i]  + dt * Xdd
        X[i+1]     = X[i]     + dt * Xdot[i]
        xdot[i+1]  = xdot[i]  + dt * xdd
        x_tmd[i+1] = x_tmd[i] + dt * xdot[i]
        T[i+1]     = T[i]     + dt

    Xddot[-1] = Xddot[-2]
    return T, X, x_tmd, Xddot


# ══════════════════════════════════════════════════════
# PARTIE B — DÉCROISSANCE LIBRE
# ══════════════════════════════════════════════════════
print("\nPARTIE A — Décroissance libre...")

t_sim = 20.0   # [s]  durée de simulation

T_ls, X_ls, _,    Add_ls = euler('libre', avec_tmd=False, X0_=X0, t_max_libre=t_sim)
T_la, X_la, xa_l, Add_la = euler('libre', avec_tmd=True,  X0_=X0, t_max_libre=t_sim)
print("  Terminé ✓")


# ══════════════════════════════════════════════════════
# DONNÉES PHYPHOX — À REMPLIR
# ══════════════════════════════════════════════════════
#
# Exporte tes données depuis Phyphox (bouton "Share" → CSV).
# Tu obtiens deux colonnes : temps [s] et accélération [m/s²].
#
# Option 1 — coller les valeurs directement ici :
#   t_phyphox     = np.array([0.0, 0.01, 0.02, ...])
#   accel_phyphox = np.array([0.1, 0.3, ...])
#
# Option 2 — charger un fichier CSV exporté par Phyphox :
#   import pandas as pd
#   df = pd.read_csv("ma_mesure.csv", sep=";")
#   t_phyphox     = df["Zeit (s)"].values       # colonne temps
#   accel_phyphox = df["Beschleunigung (m/s²)"].values  # colonne accel
#
# En attendant tes vraies mesures, on simule des données fictives
# (bruit + décroissance) pour montrer comment ça s'affichera :

np.random.seed(0)
t_phyphox_sans     = T_ls[::5]
accel_phyphox_sans = (Add_ls[::5]
                      + 0.08 * np.random.randn(len(T_ls[::5])))   # bruit ±0.08 m/s²

t_phyphox_avec     = T_la[::5]
accel_phyphox_avec = (Add_la[::5]
                      + 0.08 * np.random.randn(len(T_la[::5])))

# ══════════════════════════════════════════════════════
# PARTIE B — RÉPONSE FORCÉE
# ══════════════════════════════════════════════════════
print("PARTIE B — Réponse forcée à la résonance...")
T_fs, X_fs, _, _     = euler('force', avec_tmd=False, F0_=F0, Omega_=Omega)
T_fa, X_fa, x_fa, _ = euler('force', avec_tmd=True,  F0_=F0, Omega_=Omega)

mi = len(T_fs) // 2
amp_sans = np.max(np.abs(X_fs[mi:]))
amp_avec = np.max(np.abs(X_fa[mi:]))
reduct   = (1 - amp_avec / amp_sans) * 100
print(f"  Sans TMD : {amp_sans*1e3:.1f} mm   Avec TMD : {amp_avec*1e3:.1f} mm   → −{reduct:.0f}%")

# ── Balayage fréquentiel ───────────────────────────
print("  Balayage fréquentiel...")
n_freq  = 120
omegas  = np.linspace(0.4*omega0, 1.7*omega0, n_freq)
A_sans  = np.zeros(n_freq)
A_avec  = np.zeros(n_freq)

for i, om in enumerate(omegas):
    dt = (2*np.pi/om) / 400
    N  = 45 * 400
    T_ = np.zeros(N)

    Xs = np.zeros(N); Xds = np.zeros(N)
    Xa = np.zeros(N); Xda = np.zeros(N)
    xa = np.zeros(N); xda = np.zeros(N)

    for j in range(N - 1):
        F = F0 * np.cos(om * T_[j])

        Xdds = (-K*Xs[j] - C*Xds[j] + F) / M
        Xds[j+1] = Xds[j] + dt * Xdds
        Xs[j+1]  = Xs[j]  + dt * Xds[j]

        Fl = k*(xa[j]-Xa[j]) + c*(xda[j]-Xda[j])
        Xdda = (-K*Xa[j] - C*Xda[j] + Fl + F) / M
        xdda = (-k*(xa[j]-Xa[j]) - c*(xda[j]-Xda[j])) / m
        Xda[j+1] = Xda[j] + dt * Xdda
        Xa[j+1]  = Xa[j]  + dt * Xda[j]
        xda[j+1] = xda[j] + dt * xdda
        xa[j+1]  = xa[j]  + dt * xda[j]
        T_[j+1]  = T_[j]  + dt

    mi2 = N // 2
    A_sans[i] = np.max(np.abs(Xs[mi2:]))
    A_avec[i] = np.max(np.abs(Xa[mi2:]))

print("  Terminé ✓")


# ══════════════════════════════════════════════════════
# TRACÉS
# ══════════════════════════════════════════════════════
ROUGE = "#c0392b"
BLEU  = "#1a4fa0"
VERT  = "#1a7a45"

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
})

fig = plt.figure(figsize=(14, 16))
fig.suptitle(
    "Amortisseur de Masse Accordée (TMD) — Simulation complète\n"
    f"M = {M*1000:.0f} g,  m = {m*1000:.0f} g,  "
    f"f₀ = {f0:.2f} Hz,  L = {L_pend*100:.1f} cm",
    fontsize=13, fontweight="bold"
)

gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.35)

# ── A1 : Réponse temporelle forcée ──────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(T_fs[mi:], X_fs[mi:]*1e3, ROUGE, lw=1.3,
        label=f"Sans TMD  ({amp_sans*1e3:.1f} mm)")
ax.plot(T_fa[mi:], X_fa[mi:]*1e3, BLEU,  lw=1.3,
        label=f"Avec TMD  ({amp_avec*1e3:.1f} mm,  −{reduct:.0f}%)")
ax.set(xlabel="Temps [s]", ylabel="Déplacement X [mm]",
       title="A1 — Réponse forcée à la résonance\n(régime permanent)")
ax.legend(fontsize=9)

# ── A2 : Courbe de résonance ────────────────────────
ax = fig.add_subplot(gs[0, 1])
fn = omegas / omega0
ax.plot(fn, A_sans*1e3, ROUGE, lw=2, label="Sans TMD")
ax.plot(fn, A_avec*1e3, BLEU,  lw=2, label="Avec TMD")
ax.axvline(1.0, color="gray", lw=0.9, ls="--", alpha=0.7)
ax.set(xlabel="Fréquence normalisée  Ω/ω₀", ylabel="Amplitude [mm]",
       title="A2 — Courbe de résonance")
ax.legend(fontsize=9)
ax.set_ylim(0, None)

# ── B1 : Décroissance libre — déplacement ───────────
ax = fig.add_subplot(gs[1, :])
ax.plot(T_ls, X_ls*1e3, ROUGE, lw=1.2, alpha=0.9,
        label="Sans TMD — simulation")
ax.plot(T_la, X_la*1e3, BLEU,  lw=1.2, alpha=0.9,
        label="Avec TMD — simulation")
ax.set(xlabel="Temps [s]", ylabel="Déplacement X [mm]",
       title=f"B1 — Décroissance libre (impulsion X₀ = {X0*100:.0f} cm)\n"
             "     Avec TMD : la décroissance est beaucoup plus rapide")
ax.legend(fontsize=9)

# Annotation : temps de demi-amplitude sans TMD
tau_sans = M / C   # constante de temps approx
ax.axhline(X0*1e3 / 2, color=ROUGE, lw=0.7, ls=":", alpha=0.5)
ax.text(tau_sans * 0.7, X0*1e3/2 + 0.5,
        f"A₀/2 sans TMD", color=ROUGE, fontsize=8)

# ── B2 : Décroissance libre — ACCÉLÉRATION (Phyphox) ─
ax = fig.add_subplot(gs[2, :])
ax.plot(T_ls, Add_ls, ROUGE, lw=1.2, alpha=0.85,
        label="Sans TMD — simulation")
ax.plot(T_la, Add_la, BLEU,  lw=1.2, alpha=0.85,
        label="Avec TMD — simulation")

# Points Phyphox (fictifs pour l'instant → remplace par tes données)
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

# Annotation unité g
ax2 = ax.twinx()
ax2.set_ylim(np.array(ax.get_ylim()) / g)
ax2.set_ylabel("Accélération [g]")
ax2.spines[["top"]].set_visible(False)

plt.savefig("tmd_complet.png",
            dpi=150, bbox_inches="tight", facecolor="white")
print("\nFigure sauvegardée !")

# ══════════════════════════════════════════════════════
# RÉSUMÉ CONSOLE
# ══════════════════════════════════════════════════════
print("\n╔════════════════════════════════════════════╗")
print("║            RÉSULTATS                       ║")
print("╠════════════════════════════════════════════╣")
print(f"║  Réduction amplitude forcée  : −{reduct:.0f} %       ║")
# Temps pour que l'amplitude tombe sous 10 % de X0
seuil = 0.10 * X0
idx_s = next((i for i, v in enumerate(np.abs(X_ls)) if v < seuil and i > 10), None)
idx_a = next((i for i, v in enumerate(np.abs(X_la)) if v < seuil and i > 10), None)
if idx_s:
    print(f"║  Temps décroissance −90% sans TMD : {T_ls[idx_s]:.1f} s  ║")
if idx_a:
    print(f"║  Temps décroissance −90% avec TMD : {T_la[idx_a]:.1f} s  ║")
print(f"║  ► Longueur pendule à régler  : {L_pend*100:.2f} cm   ║")
print("╚════════════════════════════════════════════╝")

plt.show()
