#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fichier: GenPRRR.py
Générateur de trajectoire PRRR (rail X + bras RRR)
Version 1.0
"""
import numpy as np
import argparse
import sys
import RobotPRRR as robot

print("GenPRRR: Version 1.0")

parser = argparse.ArgumentParser(description="Génération de trajectoire d'un robot PRRR planaire")
parser.add_argument("-p", "--param", default="RobotPRRR.par", help="Fichier configuration (.par)")
parser.add_argument("-t", "--traj", default="Rectangle", help="Fichier trajectoire (.xy)->(.trj)")
parser.add_argument("-a", "--articulaire", type=int, default=1, help="Évitement articulaire 0, 1 ou 2")
parser.add_argument("--debug", action="store_true", help="Active les messages.")
args = parser.parse_args()

try:
    par = np.loadtxt(args.param)
except Exception as e:
    print(f"❌ Erreur de lecture de la configuration robot {args.param}: {e}")
    sys.exit(1)

if len(par) < 26:
    print(f"❌ Fichier configuration {args.param} incomplet (attendu ≥26).")
    sys.exit(1)

L = par[0:3]
base = par[7:9]

d1_dep = par[9]
d1_limits = par[10:12]
t_dep = par[12:15].reshape(3)
t_limits = par[15:21].reshape(3,2)

q = np.array([d1_dep, t_dep[0], t_dep[1], t_dep[2]], dtype=float)
limits = np.zeros((4,2))
limits[0,:] = d1_limits
limits[1:,:] = t_limits

dt = par[25]/1000.0

print(f"✅ Configuration {args.param} chargée.")
if args.debug:
    print("q_dep=", q)
    print("limits=\n", limits)
    print("dt=", dt)

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

# état initial
vmax, pas = 0.5, 20   # vmax en m/s (raisonnable)
q_list = [q.copy()]
p_current = robot.forward(q, L, base)
waypoints = np.vstack((waypoints, p_current))  # retour au départ

if args.articulaire == 0:
    i = float(input("Valeur de phi [-4:+4] x pi/4? "))
    phi = i*np.pi/4
else:
    phi = 0.0

for wp in waypoints:
    distance = np.linalg.norm(wp - p_current)
    nsteps = int(distance / (vmax*dt))
    if nsteps < 1:
        nsteps = 1

    j = pas
    for i in range(nsteps):
        alpha = (i+1)/nsteps
        p_target = (1-alpha)*p_current + alpha*wp
        q = robot.Reach(p_target, q, L, base, dt, limits, args.articulaire, phi)

        if j > 0:
            j -= 1
        else:
            q_list.append(q.copy())
            j = pas

    p_current = wp

np.savetxt(fichier_trj, np.array(q_list))
print(f"✅ Trajectoire sauvegardée dans {fichier_trj} ({len(q_list)} lignes).")
