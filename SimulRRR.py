#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: SimulRRR.py
Animation of a planar RRR robot from
- robot configuration (.par)
- joint trajectory (.trj).
Version 2.0 Batch
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import sys
import RobotRRR as robot # Cinématique du robot RRR

print("SimulRRR: Version 2.0")
# ==== Argument parser ====
parser = argparse.ArgumentParser(description="Animation d'un robot planaire RRR (version 2.0)")
parser.add_argument("-p", "--par", default="RobotRRR.par", help="Fichier de configuration robot (.par)")
parser.add_argument("-t", "--traj", default="Rectangle.trj", help="Fichier trajectoire (.trj)")
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
   
xwall, ywall = np.zeros(2), np.zeros(2)
L = par[0:3]
xmin, xmax = par[3:5]
ymin, ymax = par[5:7]
base     = par[7:9]
t_dep  = par[9:12].reshape(3)
limits = par[12:18].reshape(3,2)
xwall = par[18:20]
ywall = par[20:22]
dt   = int(par[22])
print(f"✅ Robot configuration {args.par} loaded.")
if args.debug:
    print(f"L1={L1}, L2={L2}, L3={L3}")
    print(f"x0={x0}, y0={y0}")   
    print(f"xwall[0]={xwall[0]}, ywall[0]={ywall[0]}")
    print(f"xwall[1]={xwall[1]}, ywall[1]={ywall[1]}")

# ==== Lecture trajectoire (.trj) ====
try:
    traj = np.loadtxt(args.traj)
except Exception as e:
    print(f"❌ Erreur de lecture de la trajectoire {args.traj}: {e}")
    sys.exit(1)

if traj.ndim != 2 or traj.shape[1] != 3:
    print(f"❌ Trajectoire {args.traj} doit avoir 3 colonnes (theta1, theta2, theta3).")
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
def init():              # Initialisation de l'animation
    arm.set_data([], [])
    path.set_data([], [])
    return arm, path

def update(frame):       # Répétition de l'animation
    theta = traj[frame]
    # Vérification des limites articulaires
    if not (limits[0][0] < theta[0] < limits[0][1] and
            limits[1][0] < theta[1] < limits[1][1] and
            limits[2][0] < theta[2] < limits[2][1]):
        arm.set_color("red")   # At+Hors limits -> Rouge
    else:
        arm.set_color("blue")  # Normal -> Bleu
 
    p = robot.forward_all(theta, L, base)
    arm.set_data(p[:,0], p[:,1])    # Position du robot
    path_x.append(p[3,0])  # Ajout de x au trajet
    path_y.append(p[3,1])  # Ajout de y au trajet
    path.set_data(path_x, path_y)

    return arm, path

ani = FuncAnimation(fig, update, frames=traj_len, init_func=init,
                    blit=True, interval=dt)

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title(f"Robot: {args.par}\nTrajectoire: {args.traj}")

# Save if requested
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
