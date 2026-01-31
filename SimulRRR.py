#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimulRRR_with_angles_and_ee.py
- Animation robot RRR
- Evolution articulations vs progression (%) avec double axe (d / angles)
- Evolution effecteur: x,y,phi vs progression (%)
Version 2.3
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import sys
import RobotRRR as robot

print("SimulRRR: Version 2.3 (joints + ee pose)")

# ==== Argument parser ====
parser = argparse.ArgumentParser(description="Animation robot RRR + joints + effecteur")
parser.add_argument("-p", "--par", default="RobotRRR.par", help="Fichier configuration robot (.par)")
parser.add_argument("-t", "--traj", default="Rectangle.trj", help="Fichier trajectoire (.trj)")
parser.add_argument("-o", "--output", default="", help="Fichier sortie (.gif ou .mp4), vide = rien")
parser.add_argument("--fps", type=int, default=25, help="FPS sauvegarde")
parser.add_argument("--debug", action="store_true", help="Debug print")
args = parser.parse_args()

# ==== Lecture configuration (.par) ====
try:
    par = np.loadtxt(args.par)
except Exception as e:
    print(f"❌ Erreur lecture {args.par}: {e}")
    sys.exit(1)

if len(par) < 26:
    print(f"❌ Fichier {args.par} invalide (≥26).")
    sys.exit(1)

L = par[0:3]
xmin, xmax = par[3:5]
ymin, ymax = par[5:7]
base = par[7:9]
q_dep = par[9:13].reshape(4)
limits = par[13:21].reshape(4, 2)
xwall = par[21:23]
ywall = par[23:25]
dt = int(par[25])

# ==== Lecture trajectoire (.trj) ====
try:
    traj = np.loadtxt(args.traj)
except Exception as e:
    print(f"❌ Erreur lecture {args.traj}: {e}")
    sys.exit(1)

if traj.ndim != 2 or traj.shape[1] != 4:
    print("❌ Trajectoire doit avoir 4 colonnes: (d, theta1, theta2, theta3)")
    sys.exit(1)

traj_len = len(traj)
progress = np.linspace(0.0, 100.0, traj_len)

# ==== Pré-calcul de la pose effecteur pour toute la trajectoire ====
# On calcule xe, ye et phi (orientation outil dans le plan)
xe_all = np.zeros(traj_len)
ye_all = np.zeros(traj_len)
phi_all = np.zeros(traj_len)

for k in range(traj_len):
    q = traj[k]
    p = robot.forward_all(q, L, base)     # p shape (4,2) typiquement
    xe_all[k] = p[3, 0]
    ye_all[k] = p[3, 1]
    # orientation: direction du dernier segment (joint3->EE)
    dx = p[3, 0] - p[2, 0]
    dy = p[3, 1] - p[2, 1]
    phi_all[k] = np.arctan2(dy, dx)

# (optionnel) unwrap pour éviter les sauts -pi/pi
phi_all = np.unwrap(phi_all)

# ==== Figure & axes: 3 panneaux ====
fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0])

ax_anim = fig.add_subplot(gs[0, 0])
ax_joint = fig.add_subplot(gs[0, 1])
ax_ee = fig.add_subplot(gs[0, 2])

# --- Animation axis ---
ax_anim.set_xlim(xmin, xmax)
ax_anim.set_ylim(ymin, ymax)
ax_anim.set_aspect('equal')
ax_anim.grid(True)

# --- Joint axis (double y) ---
ax_joint.set_xlim(0, 100)
ax_joint.grid(True)
ax_joint.set_xlabel("Progression sur la trajectoire (%)")
ax_d = ax_joint
ax_th = ax_joint.twinx()
ax_d.set_ylabel("d (m)")
ax_th.set_ylabel("Angles (rad)")
ax_joint.set_title("Évolution articulaire")

# --- EE axis (double y) ---
ax_ee.set_xlim(0, 100)
ax_ee.grid(True)
ax_ee.set_xlabel("Progression sur la trajectoire (%)")
ax_xy = ax_ee
ax_phi = ax_ee.twinx()
ax_xy.set_ylabel("Position (m)")
ax_phi.set_ylabel("Orientation φ (rad)")
ax_ee.set_title("Pose de l'effecteur")

# ==== Objets animation ====
wall, = ax_anim.plot(xwall, ywall, 'o-', lw=5, color="black")
arm,  = ax_anim.plot([], [], 'o-', lw=3, color="blue")
path, = ax_anim.plot([], [], 'r--', lw=1)
path_x, path_y = [], []

# ==== Données articulaires ====
d_all  = traj[:, 0]
t1_all = traj[:, 1]
t2_all = traj[:, 2]
t3_all = traj[:, 3]

# --- Fond (gris) joints ---
ax_d.plot(progress, d_all, lw=1, alpha=0.25)
ax_th.plot(progress, t1_all, lw=1, alpha=0.25)
ax_th.plot(progress, t2_all, lw=1, alpha=0.25)
ax_th.plot(progress, t3_all, lw=1, alpha=0.25)

# --- Fond (gris) EE ---
ax_xy.plot(progress, xe_all, lw=1, alpha=0.25)
ax_xy.plot(progress, ye_all, lw=1, alpha=0.25)
ax_phi.plot(progress, phi_all, lw=1, alpha=0.25)

# --- Lignes progressives joints ---
d_line,  = ax_d.plot([], [],  lw=2, label=r"$d$")
t1_line, = ax_th.plot([], [], lw=2, label=r"$\theta_1$")
t2_line, = ax_th.plot([], [], lw=2, label=r"$\theta_2$")
t3_line, = ax_th.plot([], [], lw=2, label=r"$\theta_3$")

# --- Lignes progressives EE ---
xe_line,  = ax_xy.plot([], [], lw=2, label=r"$x_e$")
ye_line,  = ax_xy.plot([], [], lw=2, label=r"$y_e$")
phi_line, = ax_phi.plot([], [], lw=2, label=r"$\phi$")

# Curseurs verticaux
cursor_joint = ax_joint.axvline(0.0, lw=2, alpha=0.7)
cursor_ee    = ax_ee.axvline(0.0, lw=2, alpha=0.7)

# Légendes combinées
lines_joint = [d_line, t1_line, t2_line, t3_line]
ax_joint.legend(lines_joint, [l.get_label() for l in lines_joint], loc="best")

lines_ee = [xe_line, ye_line, phi_line]
ax_ee.legend(lines_ee, [l.get_label() for l in lines_ee], loc="upper left")

# ==== Animation ====
def init():
    arm.set_data([], [])
    path.set_data([], [])

    d_line.set_data([], [])
    t1_line.set_data([], [])
    t2_line.set_data([], [])
    t3_line.set_data([], [])

    xe_line.set_data([], [])
    ye_line.set_data([], [])
    phi_line.set_data([], [])

    cursor_joint.set_xdata([0.0])
    cursor_ee.set_xdata([0.0])

    return [
        arm, path,
        d_line, t1_line, t2_line, t3_line, cursor_joint,
        xe_line, ye_line, phi_line, cursor_ee
    ]

def update(frame):
    q = traj[frame]  # [d, t1, t2, t3]

    # limites articulaires
    if not all(limits[i, 0] <= q[i] <= limits[i, 1] for i in range(4)):
        arm.set_color("red")
    else:
        arm.set_color("blue")

    # robot + path
    p = robot.forward_all(q, L, base)
    arm.set_data(p[:, 0], p[:, 1])
    path_x.append(p[3, 0])
    path_y.append(p[3, 1])
    path.set_data(path_x, path_y)

    pr = progress[:frame + 1]

    # joints
    d_line.set_data(pr, d_all[:frame + 1])
    t1_line.set_data(pr, t1_all[:frame + 1])
    t2_line.set_data(pr, t2_all[:frame + 1])
    t3_line.set_data(pr, t3_all[:frame + 1])

    # effecteur
    xe_line.set_data(pr, xe_all[:frame + 1])
    ye_line.set_data(pr, ye_all[:frame + 1])
    phi_line.set_data(pr, phi_all[:frame + 1])

    # curseurs
    cursor_joint.set_xdata([progress[frame]])
    cursor_ee.set_xdata([progress[frame]])

    return [
        arm, path,
        d_line, t1_line, t2_line, t3_line, cursor_joint,
        xe_line, ye_line, phi_line, cursor_ee
    ]

ani = FuncAnimation(fig, update, frames=traj_len, init_func=init,
                    blit=True, interval=dt)

ax_anim.set_title(f"Robot: {args.par}\nTrajectoire: {args.traj}")
ax_anim.set_xlabel("x (m)")
ax_anim.set_ylabel("y (m)")

# ==== Save if requested ====
if args.output:
    if args.output.endswith(".gif"):
        print(f"✅ Enregistrement {args.output} ...")
        ani.save(args.output, writer="pillow", fps=args.fps)
    elif args.output.endswith(".mp4"):
        print(f"✅ Enregistrement {args.output} ...")
        ani.save(args.output, writer="ffmpeg", fps=args.fps)
    else:
        print("❌ Fichier .gif ou .mp4 seulement")

plt.tight_layout()
plt.show()
