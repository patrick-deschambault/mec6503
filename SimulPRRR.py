#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: SimulPRRR.py
Animation robot PRRR (rail X + bras RRR)
Version 1.0
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import sys
import RobotPRRR as robot

print("SimulPRRR: Version 1.0")

parser = argparse.ArgumentParser(description="Animation d'un robot planaire PRRR")
parser.add_argument("-p", "--par", default="RobotPRRR.par", help="Fichier de configuration robot (.par)")
parser.add_argument("-t", "--traj", default="Rectangle1.trj", help="Fichier trajectoire (.trj)")
parser.add_argument("--debug", action="store_true", help="Messages")
args = parser.parse_args()

try:
    par = np.loadtxt(args.par)
except Exception as e:
    print(f"❌ Erreur lecture {args.par}: {e}")
    sys.exit(1)

if len(par) < 26:
    print(f"❌ Fichier {args.par} incomplet (≥26).")
    sys.exit(1)

L = par[0:3]
xmin, xmax = par[3:5]
ymin, ymax = par[5:7]
base = par[7:9]

d1_limits = par[10:12]
t_limits = par[15:21].reshape(3,2)
limits = np.zeros((4,2))
limits[0,:] = d1_limits
limits[1:,:] = t_limits

xwall = par[21:23]
ywall = par[23:25]
dt_ms = int(par[25])

traj = np.loadtxt(args.traj)
if traj.ndim != 2 or traj.shape[1] != 4:
    print(f"❌ Trajectoire {args.traj} doit avoir 4 colonnes (d1,t1,t2,t3).")
    sys.exit(1)

fig, ax = plt.subplots()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal')
ax.grid(True)

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

    in_limits = True
    for j in range(4):
        if not (limits[j,0] <= q[j] <= limits[j,1]):
            in_limits = False
            break
    arm.set_color("blue" if in_limits else "red")

    p = robot.forward_all(q, L, base)
    arm.set_data(p[:,0], p[:,1])

    path_x.append(p[-1,0])
    path_y.append(p[-1,1])
    path.set_data(path_x, path_y)
    return arm, path

ani = FuncAnimation(fig, update, frames=len(traj), init_func=init,
                    blit=True, interval=dt_ms)

plt.title(f"Robot: {args.par}\nTrajectoire: {args.traj}")
plt.show()
