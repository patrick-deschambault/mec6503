#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fichier: GenRRR.py
Générateur de trajectoire pour robot planaire RRR
- Lecture du fichier paramètre (.par)
- Lecture du fichier des points de passages (.xy)
- Écriture du fichier trajectoire (.trj)
- Stratégie d'évitement:
    -a 0 : Orientation phi constante
    -a 1 : Position à norme-minimale
    -a 2 : Position à h vers le centre articulaire
Version 2.0
"""
import numpy as np
import argparse
import sys
import RobotRRR as robot	 # Cinématique du robot RRR

print("GenRRR: Version 2.0")
# ==== Traitement en batch ====
parser = argparse.ArgumentParser(description="Génération de trajectoire d'un robot RRR planaire")
parser.add_argument("-p", "--param", default="RobotRRR.par", help="Fichier configuration (.par)")
parser.add_argument("-t", "--traj", default="Rectangle", help="Fichier trajectoire (.xy)->(.trj)")
parser.add_argument("-a", "--articulaire", type=int, default=0, help="Évitement articulaire 0, 1 ou 2")
parser.add_argument("--debug", action="store_true", help="Active les messages.")
args = parser.parse_args()

# ==== Lecture de la configuration robot (.par) ====
try:
    par = np.loadtxt(args.param)
except Exception as e:
    print(f"❌ Erreur de lecture de la configuration robot {args.param}: {e}")
    sys.exit(1)

if len(par) < 23:
    print(f"❌ Fichier configuration {args.param} incomplet (attendu ≥23).")
    sys.exit(1)

L = par[0:3] # L1 L2 L3
base = par[7:9] 
t_dep = par[9:13].reshape(4) # d, q1, q2, q3
limits = par[13:21].reshape(4,2) # limits for d, q1, q2, q3
dt   = par[22]/1000 # Convert ms to s
print(f"✅ Configuration {args.param} à {len(par)} paramètres chargée.")
if args.debug: print("par=", par)

# ==== Lecture des points de passage (.xy) ====
fichier_xy  = args.traj + ".xy"
fichier_trj = args.traj + f"{args.articulaire}.trj"
   
try:
    waypoints = np.loadtxt(fichier_xy)
except Exception as e:
    print(f"❌ Erreur de lecture de la trajectoire {fichier_xy}: {e}")
    sys.exit(1)

if waypoints.ndim != 2 or waypoints.shape[1] != 2:
    print(f"❌ Fichier trajectoire {fichier_xy} doit avoir 2 colonnes (x,y).")
    sys.exit(1)

print(f"✅ Trajectoire {fichier_xy} à {len(waypoints)} points chargée.")
if args.debug: print("waypoints=", waypoints)
        
# ==== État initial ====
vmax, pas = 1.0, 20 # Vitesse modifiée
theta = t_dep
q_list = [theta.copy()]
p_current = robot.forward(theta, L, base)
waypoints = np.vstack((waypoints,p_current)) # Retour au départ
if args.debug:
    print(f"Position de départ {t_dep} -> {p_current} actuelle. ")
    if args.articulaire==1: print("Stratégie de saturation articulaire.")
    if args.articulaire==2:
        print("Stratégie de projection sur le noyau de J.")

if args.articulaire==0: # Choisir -pi <= phi <= + pi
    i = float(input("Valeur de phi [-4:+4] x pi/4?"))
    phi = i*np.pi/4
else:
    phi=0
     
# ==== Suivi des segments ====
for wp in waypoints:
    distance = np.linalg.norm(wp - p_current)
    nsteps = int(distance*vmax/dt)
    if nsteps < 1: nsteps = 1
    print(f"nsteps={nsteps}")
    j = pas
    for i in range(nsteps):
        alpha = (i+1)/nsteps
        p_target = (1-alpha)*p_current + alpha*wp
        theta = robot.Reach(p_target,theta,L,base,dt,limits,args.articulaire,phi)
        # ==== Enregistre 1 point sur 10 dans le fichier
        if j>0: 
            j = j-1
        else:
            q_list.append(theta.copy())
            j = pas
    p_current = wp
    print(f"p={p_current}, theta={theta}")

# Sauvegarde
np.savetxt(fichier_trj, np.array(q_list))
print(f"✅ Trajectoire sauvegardée dans {fichier_trj} ({len(q_list)} lignes).")