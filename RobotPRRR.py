#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fichier: RobotPRRR.py
Cinématique du robot PRRR
- forward_all() et forward()
- jacobian()
- update_with_limits() et hit_limit()
- Stratégie d'évitement Reach avec:
    -a 0 : Orientation phi constante
    -a 1 : Position à norme-minimale
    -a 2 : Position à h vers le centre articulaire
Version 1.0
"""
import numpy as np

# Cinématique direct: Position des 4 points
def forward_all(q, L, base):
    x_base, y_base = base
    d0, t1, t2, t3 = q

    # base mobile
    x0 = x_base + d0
    y0 = y_base

    # lien 1
    x1 = x0 + L[0]*np.cos(t1)
    y1 = y0 + L[0]*np.sin(t1)
    # lien 2
    x2 = x1 + L[1]*np.cos(t1+t2)
    y2 = y1 + L[1]*np.sin(t1+t2)
    # lien 3
    x3 = x2 + L[2]*np.cos(t1+t2+t3)
    y3 = y2 + L[2]*np.sin(t1+t2+t3)

    return np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])

# Cinématique directe : Bout seulement
def forward(q, L, base):
    return forward_all(q, L, base)[-1]

# Construire la matrice Jacobienne 2x4
def jacobian(q, L):
    d0, t1, t2, t3 = q
    l1, l2, l3 = L
    s1, c1 = np.sin(t1), np.cos(t1)
    s12, c12 = np.sin(t1+t2), np.cos(t1+t2)
    s123, c123 = np.sin(t1+t2+t3), np.cos(t1+t2+t3)

    J = np.zeros((2,4))

    # Joint P (translation)
    J[0,0] = 1.0
    J[1,0] = 0.0

    # Joint R1
    J[0,1] = -l1*s1 - l2*s12 - l3*s123
    J[1,1] =  l1*c1 + l2*c12 + l3*c123

    # Joint R2
    J[0,2] = -l2*s12 - l3*s123
    J[1,2] =  l2*c12 + l3*c123

    # Joint R3
    J[0,3] = -l3*s123
    J[1,3] =  l3*c123

    return J

# Mise à jour avec respect des limites
def update_with_limits(q, dq, dt, limits):
    new = q + dt*dq
    for j in range(4):
        new[j] = np.clip(new[j], limits[j,0], limits[j,1])
    return new

def hit_limit(q, limits):
    return np.any((q <= limits[:,0]) | (q >= limits[:,1]))

# Cinématique inverse (PRRR)
def inverse_PRRR(pf, phi, L, base, d0=0):
    # Calculer IK des 3 rotatifs
    x0, y0 = base
    theta1, theta2, theta3 = inverse_RRR(pf, phi, L, base)
    # Ajouter d0 en premier
    return [d0, theta1, theta2, theta3]

# Ancienne inverse RRR utilisée à l’intérieur
def inverse_RRR(pf, phi, L, base):
    x0, y0 = base    
    L1, L2, L3 = L
    
    # calcul du poignet
    wx = pf[0] - L3 * np.cos(phi) - x0
    wy = pf[1] - L3 * np.sin(phi) - y0

    # IK 2R pour atteindre le poignet
    D = (wx**2 + wy**2 - L1**2 - L2**2) / (2*L1*L2)
    D = np.clip(D, -1.0, 1.0)

    theta2 = np.arctan2(np.sqrt(1 - D**2), D)

    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)
    theta1 = np.arctan2(wy, wx) - np.arctan2(k2, k1)

    # angle du dernier lien
    theta3 = phi - theta1 - theta2
    return [theta1, theta2, theta3]

# Atteindre pf à partir de theta selon 3 stratégies (a = 0, 1 ou 2)
def Reach_PRRR(pf, q, L, base, dt, limits, a, phi=0, d0_fixed=None):
    n, e, d = 25, 1, 0.75
    q_mid = np.array([ (lim[0]+lim[1])/2 for lim in limits ])
    q_rng = np.array([ 1/(lim[1]-lim[0]) if lim[1]-lim[0] != 0 else 1.0 for lim in limits ])
    I = np.eye(4)
    W = 2.0*np.diag(q_rng)

    if a==0:
        q = inverse_PRRR(pf, phi, L, base, d0_fixed if d0_fixed is not None else 0)
    else:
        p = forward(q, L, base)
        while n>0 and e>0.00001:
            dp = d*(pf - p)/dt
            J = jacobian(q, L)
            J_pinv = np.linalg.pinv(J)
            dq = J_pinv @ dp

            if d0_fixed is not None:
                dq[0] = 0  # verrouiller le rail si nécessaire

            if a==2:
                h = W @ (q_mid - q)
                dq2 = d*(I - J_pinv @ J) @ h
                dq = dq + dq2

            q = update_with_limits(q, dq, dt, limits)
            p = forward(q, L, base)
            e = np.linalg.norm(pf - p)
            n -= 1

    return q
