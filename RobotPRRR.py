#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fichier: RobotPRRR.py
Cinématique d'un robot planaire PRRR
- forward_all() et forward()
- jacobian()
- update_with_limits() et hit_limit()
- Stratégie d'évitement Reach avec:
    -a 0 : Orientation phi constante (IK "directe" + choix de p minimal)
    -a 1 : Suivi Jacobien (sans évitement)
    -a 2 : Suivi Jacobien + évitement vers le centre des limites (nullspace)
Convention: PRRR = [p, t1, t2, t3]
- p : joint prismatique (translation le long de +x du monde)
- t1,t2,t3 : joints rotatifs
Version 1.0
"""
import numpy as np

# ===== Cinématique directe: Position des 5 points (base, après P, puis 3 articulations) =====
def forward_all(theta, L, base):
    x0, y0 = base
    p, t1, t2, t3 = theta
    l1, l2, l3 = L

    # point après prisme (translation en x)
    xP = x0 + p
    yP = y0

    x1 = xP + l1 * np.cos(t1)
    y1 = yP + l1 * np.sin(t1)

    x2 = x1 + l2 * np.cos(t1 + t2)
    y2 = y1 + l2 * np.sin(t1 + t2)

    x3 = x2 + l3 * np.cos(t1 + t2 + t3)
    y3 = y2 + l3 * np.sin(t1 + t2 + t3)

    # [base] -> [après P] -> [après R1] -> [après R2] -> [effecteur]
    return np.array([[x0, y0], [xP, yP], [x1, y1], [x2, y2], [x3, y3]])

def forward(theta, L, base):
    return forward_all(theta, L, base)[-1]

# ===== Jacobienne 2x4 (x,y) par rapport à [p,t1,t2,t3] =====
def jacobian(theta, L):
    p, t1, t2, t3 = theta
    l1, l2, l3 = L

    s1,  c1  = np.sin(t1),          np.cos(t1)
    s12, c12 = np.sin(t1 + t2),     np.cos(t1 + t2)
    s123,c123= np.sin(t1 + t2 + t3),np.cos(t1 + t2 + t3)

    J = np.zeros((2, 4))

    # d(x,y)/dp : prisme le long de x monde
    J[0, 0] = 1.0
    J[1, 0] = 0.0

    # colonnes RRR identiques au cas 3R
    J[0, 1] = -l1*s1  - l2*s12  - l3*s123
    J[0, 2] =          - l2*s12  - l3*s123
    J[0, 3] =                    - l3*s123

    J[1, 1] =  l1*c1  + l2*c12  + l3*c123
    J[1, 2] =           l2*c12  + l3*c123
    J[1, 3] =                     l3*c123

    return J

# ===== Limites =====
def update_with_limits(theta, dtheta, dt, limits):
    new = np.array(theta, dtype=float) + dt * np.array(dtheta, dtype=float)
    for j in range(4):
        new[j] = np.clip(new[j], limits[j, 0], limits[j, 1])
    return new

def hit_limit(theta, limits):
    th = np.array(theta, dtype=float)
    return np.any((th <= limits[:, 0]) | (th >= limits[:, 1]))

# ===== IK (simplifiée) à orientation constante phi =====
def inverse(pf, phi, L, base, p_limits=None):
    """
    Retourne [p, t1, t2, t3] pour atteindre pf avec orientation phi.
    On choisit p (translation en x) "minimal" (|p| petit) tout en rendant l'IK 2R réalisable.
    """
    x0, y0 = base
    l1, l2, l3 = L

    # position du "poignet" (avant le dernier lien l3) en monde
    wxw = pf[0] - l3 * np.cos(phi)
    wyw = pf[1] - l3 * np.sin(phi)

    dy = wyw - y0
    rmin = abs(l1 - l2)
    rmax = (l1 + l2)

    # condition minimale: |dy| <= rmax sinon impossible même en bougeant p
    if abs(dy) > rmax + 1e-12:
        raise ValueError("IK impossible: cible trop loin en y (|dy| > L1+L2).")

    # dx = (wxw - (x0+p)) ; on peut choisir p pour rendre r dans [rmin, rmax]
    dx0 = wxw - x0  # correspond à p=0

    # dx^2 doit être dans [rmin^2 - dy^2, rmax^2 - dy^2]
    lo2 = max(0.0, rmin**2 - dy**2)
    hi2 = max(0.0, rmax**2 - dy**2)

    dx_lo = np.sqrt(lo2)
    dx_hi = np.sqrt(hi2)

    # Choisir dx le plus proche de dx0 mais avec |dx| dans [dx_lo, dx_hi]
    abs_dx0 = abs(dx0)
    if abs_dx0 < dx_lo:
        abs_dx = dx_lo
    elif abs_dx0 > dx_hi:
        abs_dx = dx_hi
    else:
        abs_dx = abs_dx0

    sign = 1.0 if dx0 >= 0 else -1.0
    dx = sign * abs_dx

    # donc p = dx0 - dx
    p = dx0 - dx

    if p_limits is not None:
        p = float(np.clip(p, p_limits[0], p_limits[1]))
        # recompute dx with clipped p
        dx = wxw - (x0 + p)

    # maintenant IK 2R sur le poignet avec base (x0+p, y0)
    D = (dx**2 + dy**2 - l1**2 - l2**2) / (2.0 * l1 * l2)
    D = float(np.clip(D, -1.0, 1.0))  # robustesse numérique

    t2 = np.arctan2(np.sqrt(1.0 - D**2), D)  # "coude haut"
    k1 = l1 + l2 * np.cos(t2)
    k2 = l2 * np.sin(t2)
    t1 = np.arctan2(dy, dx) - np.arctan2(k2, k1)

    t3 = phi - t1 - t2
    return [p, t1, t2, t3]

# ===== Reach: atteindre pf à partir de theta selon 3 stratégies =====
def Reach(pf, theta, L, base, dt, limits, a, phi=0.0):
    n, e, d = 25, 1.0, 0.75

    theta = np.array(theta, dtype=float)
    limits = np.array(limits, dtype=float)

    theta_mid = np.array([(lim[0] + lim[1]) / 2.0 for lim in limits])
    theta_rng = np.array([1.0 / (lim[1] - lim[0]) for lim in limits])

    I = np.eye(4)
    W = 2.0 * np.diag(theta_rng)

    if a == 0:
        # IK orientation constante phi + choix p minimal
        p_limits = limits[0]
        theta = np.array(inverse(pf, phi, L, base, p_limits=p_limits), dtype=float)
        # respecter aussi les limites des angles
        for j in range(4):
            theta[j] = np.clip(theta[j], limits[j, 0], limits[j, 1])
    else:
        # suivi Jacobien
        p = forward(theta, L, base)
        while n > 0 and e > 1e-5:
            dp = d * (np.array(pf, dtype=float) - p) / dt
            J = jacobian(theta, L)
            J_pinv = np.linalg.pinv(J)  # 4x2
            dtheta = J_pinv @ dp

            if a == 2:
                h = W @ (theta_mid - theta)
                dtheta2 = d * (I - J_pinv @ J) @ h
                dtheta = dtheta + dtheta2

            theta = update_with_limits(theta, dtheta, dt, limits)
            p = forward(theta, L, base)
            e = np.linalg.norm(np.array(pf, dtype=float) - p)
            n -= 1

    return theta
