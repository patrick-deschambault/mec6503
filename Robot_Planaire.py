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

print("Définition d'un robot 2D planaire RR")
# 1. Définir les symboles algébriques
q1, q2 = sp.symbols('q1 q2')  # Variable commandées
L1, L2 = sp.symbols('L1 L2')  # Longueurs fixes

# 2. Créer le modèle cinématique du robot
T01 = Matrice_2D(q1, L1)  # Matrice 2D homogène 3x3
T12 = Matrice_2D(q2, L2)  # Matrice 2D homogène 3x3

# Multipier les matrices 2D homogènes
T02 = T01 * T12  # Matrice 2D homogène 3x3
pos = T02[:2, 2] # Vecteur position 2x1

# 3. Dériver symboliquement la Jacobienne par rapport à q1, q2
J = pos.jacobian([q1, q2])  # Matrice 2x2

print("\nCinématique:")
sp.pprint(pos)
print("\nJacobienne:")
sp.pprint(J)

# 4. Créer des fonctions de calcul numérique rapide
# Inclure toutes les variables symboliques comme argument
pos_func = sp.lambdify((q1, q2, L1, L2), pos, 'numpy')
jac_func = sp.lambdify((q1, q2, L1, L2), J, 'numpy')

# 5. Évaluer les fonctions à des valeurs numériques spécifiques
# afin de valider les équations de position et la Jacobienne
q_vals = [np.pi/2, -np.pi/2]  # Angle en radian
l_vals = [2.0, 1.5]           # Longueur en m

# Utiliser l'opérateur * pour déballer les listes en arguments
current_pos = pos_func(*q_vals, *l_vals)
current_jac = jac_func(*q_vals, *l_vals)

print("\nPosition:\n", current_pos)
print("\nJacobienne:\n", current_jac)

