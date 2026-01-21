#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fichier: Robot_Planaire.py
Cours: MEC1315 Ti en ingénierie
@author: Luc Baron
"""
import sympy as sp
import numpy as np

# Matrice 2D homogène: rotation theta, puis translation en x de a
def Matrice_2D(theta, a):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta), a*sp.sin(theta)],
        [            0,              0,               1]
    ])

def Trans2D(x, y):
    return sp.Matrix([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ])

print("Définition d'un robot 2D planaire RR")
# 1. Définir les symboles algébriques
d, q1, q2, q3 = sp.symbols('d q1 q2 q3', real=True)  # Variable commandées
L1, L2, L3 = sp.symbols('L1 L2 L3', positive=True, real=True)  # Longueurs fixes
x0, y0 = sp.symbols('x0 y0', real=True)  # Base fixe

# Système de transformation avec la base et le rail.
T0B = Trans2D(x0, y0)
TBP = Trans2D(d, 0)

# 2. Créer le modèle cinématique du robot
T01 = Matrice_2D(q1, L1)  # Matrice 2D homogène 3x3
T12 = Matrice_2D(q2, L2)  # Matrice 2D homogène 3x3
T23 = Matrice_2D(q3, L3)  # Matrice 2D homogène 3x3

# On vient obtenir la transformation complète du dernier lien
T03 = T0B * TBP * T01 * T12 * T23

# Multipier les matrices 2D homogènes
pos = T03[:2, 2]
phi = sp.Matrix([q1 + q2 + q3]) # Orientation de l'effecteur par rapport a la base

# 3. Dériver symboliquement la Jacobienne par rapport à d, q1, q2 et q3
J = pos.jacobian([d, q1, q2, q3])  # Matrice 2x2

print("\nCinématique:")
sp.pprint(pos)
print("\nJacobienne:")
sp.pprint(J)

# 4. Créer des fonctions de calcul numérique rapide
# Inclure toutes les variables symboliques comme argument
pos_func = sp.lambdify((d, q1, q2, q3, L1, L2, L3, x0, y0), pos, 'numpy')
jac_func = sp.lambdify((d, q1, q2, q3, L1, L2, L3, x0, y0), J, 'numpy')

# 5. Évaluer les fonctions à des valeurs numériques spécifiques
# afin de valider les équations de position et la Jacobienne
q_vals = [0.1, np.pi/2, -np.pi/2, np.pi/2]  # Distance du joint prismatique avec les Angles en radian
l_vals = [2.0, 1.5, 1.5]                    # Longueur en m pour L1, L2 et L3
base_vals = [0.0, 0.0]  # Position de la base x0, y0

# Utiliser l'opérateur * pour déballer les listes en arguments
current_pos = pos_func(*q_vals, *l_vals, *base_vals)
current_jac = jac_func(*q_vals, *l_vals, *base_vals)

print("\nPosition:\n", current_pos)
print("\nJacobienne:\n", current_jac)

