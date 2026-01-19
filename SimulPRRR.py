#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: SimulPRRR.py
Animation of a planar PRRR robot from
- robot configuration (.par)
- joint trajectory (.trj).
Version 1.0 Batch
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import sys
import RobotPRRR as robot  # Cinématique du robot PRRR

print("SimulPRRR: Version 1.0")

# ==== Argument parser ====
parser = argparse.ArgumentParser(description="Animation d'un robot planaire PRRR")
parser.add_argument("-p", "--par", default="RobotPRRR.par", help="Fichier de configuration robot (.par)")
parser.add_argument("-t", "--traj", default="Droite0.trj", help="Fichier trajectoire (.trj)")
parser.add_argument("-o", "--output", default="", help="Fichier image (gif or mp4). Vide = pas d'enregistrement")
parser.add_argument("--fps", type=int, default=25, help="Image par seconde pour l'enregistrement")
parser.add_argument("--debug", action="store_true", help="Messages textes enrichies")
args = parser.parse_args()

# ==== Lecture configuration (.par) ====
try:
    par = np.loadtxt(args.par)
except Exception as e:
    print(f"❌ Erreur de lecture de la configuration robot: {args.par}: {e}")
    sys.exit(1)

if len(par) < 23:
    print(f"❌ Fichier {args.par} ne contient pas suffisament de paramètres (≥23).")
    sys.exit(1)

# Paramètres des 3 rotatifs + base
L = par[0:3]
xmin, xmax = par[3:5]
ymin, ymax = par[5:7]
base = par[7:9]
t_dep = par[9:12].reshape(3)
d0_dep = 0.0
q_dep = np.hstack(([d0_dep], t_dep))  # état initial PRRR
limits = np.vstack(([ [0.0, 0.0] ], par[12:18].reshape(3,2)))
xwall = par[18:20]
ywall = par[20:22]
dt = int(par[22])
print(f"✅ Robot configuration {args.par} loaded.")

# ==== Lecture trajectoire (.trj) ====
try:
    traj = np.loadtxt(args.traj)
except Exception as e:
    print(f"❌ Erreur de lecture de la trajectoire {args.traj}: {e}")
    sys.exit(1)

if traj.ndim != 2 or traj.shape[1] != 4:
    print(f"❌ Trajectoire {args.traj} doit avoir 4 colonnes (d0, theta1, theta2, theta3).")
    sys.exit(1)

traj_len = len(traj)
print(f"✅ Trajectoire {args.traj} chargée: {traj_len} pas.")

# ==== Paramètres de traçage ====
fig, ax = plt.subplots()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal')
ax.grid(True)

# ==== Objets de la scène ====
wall, = ax.plot(xwall, ywall, 'o-', lw=5, color="black")
arm,  = ax.plot([], [], 'o-', lw=3, color="blue")
path, = ax.plot([], [], 'r--', lw=1)
path_x, path_y = [], []

def init():
    arm.set_data([], [])
    path.set_data([], [])
    return arm, path

def update(frame):
    q = traj[frame]
    # Vérification des limites
    if not (limits[0][0] <= q[0] <= limits[0][1] and
            limits[1][0] <= q[1] <= limits[1][1] and
            limits[2][0] <= q[2] <= limits[2][1] and
            limits[3][0] <= q[3] <= limits[3][1]):
        arm.set_color("red")
    else:
        arm.set_color("blue")

    p = robot.forward_all(q, L, base)
    arm.set_data(p[:,0], p[:,1])
    path_x.append(p[3,0])
    path_y.append(p[3,1])
    path.set_data(path_x, path_y)
    return arm, path

ani = FuncAnimation(fig, update, frames=traj_len, init_func=init,
                    blit=True, interval=dt)

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title(f"Robot: {args.par}\nTrajectoire: {args.traj}")

# Sauvegarde
if args.output:
    if args.output.endswith(".gif"):
        print(f"✅ Enregistrement {args.output} ...")
        ani.save(args.output, writer="pillow", fps=args.fps)
    elif args.output.endswith(".mp4"):
        print(f"✅ Enregistrement {args.output} ...")
        ani.save(args.output, writer="ffmpeg", fps=args.fps)
    else:
        print("❌ Fichier .gif ou .mp4 seulement")

plt.show()
