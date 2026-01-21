#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fichier: RobotRRR.py
Cinématique du robot RRR
- forward_all() et forward()
- jacobian()
- update_with_limits() et hit_limit()
- Stratégie d'évitement Reach avec:
    -a 0 : Orientation phi constante
    -a 1 : Position à norme-minimale
    -a 2 : Position à h vers le centre articulaire
Version 2.0
"""
import numpy as np
import argparse
import sys

# Cinématique direct: Position des 4 points
def forward_all(q, L, base):
    x0, y0 = base
    d, q1, q2, q3 = q
    
    L1, L2, L3 = L
    
    # On suppose ici que d ne s'applique qu'en x (pour l'instant).
    x0_p = x0 + d
    y0_p = y0
    
    # Bout du lien 1, on doit offset les coordonnees prismatique ici.
    x1 = x0_p + L1*np.cos(q1)
    y1 = y0_p + L1*np.sin(q1)
    
    x2 = x1 + L2*np.cos(q1+q2)
    y2 = y1 + L2*np.sin(q1+q2)
    
    x3 = x2 + L3*np.cos(q1+q2+q3)
    y3 = y2 + L3*np.sin(q1+q2+q3)
    
    return np.array([[x0_p,y0_p],[x1,y1],[x2,y2],[x3,y3]])

# Cinématique direct: Position du bout seulement
def forward(q, L, base):
    return forward_all(q, L, base)[-1]

# Construire la matrice Jacobienne 2x3
def jacobian(q, L):
    d, q1, q2, q3 = q
    l1, l2, l3 = L
    
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1+q2), np.cos(q1+q2)
    s123, c123 = np.sin(q1+q2+q3), np.cos(q1+q2+q3)

    J = np.zeros((2,4))
    
    # La contribution est d'ajouter cette colonne uniquement.
    J[0,0] = 1.0
    J[1,0] = 0.0
    
    J[0,1] = -l1*s1 - l2*s12 - l3*s123
    J[0,2] = -l2*s12 - l3*s123
    J[0,3] = -l3*s123
    
    J[1,1] =  l1*c1 + l2*c12 + l3*c123
    J[1,2] =  l2*c12 + l3*c123
    J[1,3] =  l3*c123
    return J

# Mise à jour avec respect des limites
def update_with_limits(theta, dtheta, dt, limits):
    new = theta + dt*dtheta
    for j in range(4):
        new[j] = np.clip(new[j], limits[j,0], limits[j,1])
    return new

def hit_limit(theta, limits):
    return np.any((theta <= limits[:,0]) | (theta >= limits[:,1]))


# Cinématique inverse (simplifiée)
def inverse(pf, phi, L, base):
    x0, y0 = base    
    L1, L2, L3 = L
    
    # calcul du poignet
    wx = pf[0] - L3 * np.cos(phi) - x0
    wy = pf[1] - L3 * np.sin(phi) - y0

    # IK 2R pour atteindre le poignet
    D = (wx**2 + wy**2 - L1**2 - L2**2) / (2*L1*L2)
    theta2 = np.arctan2(np.sqrt(1 - D**2), D)

    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)
    theta1 = np.arctan2(wy, wx) - np.arctan2(k2, k1)

    # angle du dernier lien
    theta3 = phi - theta1 - theta2
    return [theta1, theta2, theta3]

# Atteindre pf à partir de theta selon 3 stratégies (a = 0, 1 ou 2)
def Reach(pf,theta,L,base,dt,limits,a,phi=0):
    n, e, d = 25, 1, 0.75
    theta_mid = np.array([ (lim[0]+lim[1])/2 for lim in limits ])
    theta_rng = np.array([ 1/(lim[1]-lim[0]) for lim in limits ])
    I = np.eye(3)
    W = 2.0*np.diag(theta_rng)

    if a==0:  # Cinématique inverse à orientation constante phi
        theta = inverse(pf,phi,L,base) # TODO
    else:     # Petit déplacements avec la Jacobienne
        p = forward(theta, L, base)
        while n>0 and e>0.00001:
            dp = d*(pf - p) / dt # Position controlée. On impose une vitesse cartésienne vers le point désiré.
            J = jacobian(theta, L)
            J_pinv = np.linalg.pinv(J)
            dtheta = J_pinv @ dp  # Suivre le trajet sans évitement
            
            if a==2:
                h = W @ (theta_mid - theta)
                dtheta2 = d * (I - J_pinv @ J) @ h  # avec évitement
                if np.linalg.norm(J @ dtheta2)>0.00001 : print(dtheta2)
                dtheta = dtheta + dtheta2

            theta = update_with_limits(theta, dtheta, dt, limits)
            p = forward(theta, L, base)        
            e = np.linalg.norm(pf-p)
            n = n - 1
    return theta
