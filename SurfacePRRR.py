#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SurfacePRRR.py

Affiche 11 courbes de selfmotion pour un robot PRRR
le long d'une ligne droite entre deux waypoints :
- waypoint 1
- 9 points intermédiaires
- waypoint 2

Seule la selfmotion principale (+1) est tracée pour lisibilité.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import RobotPRRR as robot

# ---- SELFMOTION POUR UN POINT FIXE ----
def SelfMotionPoint(q0, pf, L, base, limits,
                    direction=+1, dt=0.05,
                    g_null=0.25, g_task=0.15,
                    max_iter=500, eps=1e-7):

    q = q0.copy()
    traj = [q.copy()]
    I = np.eye(4)  # 4 DOF pour PRRR

    # direction vers limite haute ou basse
    target = limits[:,1] if direction == 1 else limits[:,0]

    for _ in range(max_iter):

        if robot.hit_limit(q, limits):
            break

        p = robot.forward(q, L, base)
        J = robot.jacobian(q, L)
        Jp = np.linalg.pinv(J)

        # Maintenir la position
        dp = pf - p
        dtheta_task = g_task * (Jp @ dp)

        # Selfmotion = projection sur le noyau
        h = target - q
        N = I - Jp @ J
        dtheta_null = g_null * (N @ h)

        if np.linalg.norm(dtheta_null) < eps:
            break

        dq = dtheta_task + dtheta_null
        q = robot.update_with_limits(q, dq, dt, limits)
        traj.append(q.copy())

    return np.array(traj)

# ---- MAIN SCRIPT ----
if __name__ == "__main__":
    print("SurfacePRRR: Version 1.0")

    # Charger RobotPRRR.par
    par = np.loadtxt("RobotPRRR.par")
    L = par[0:3]
    base = par[7:9]
    d0_dep = 0.0
    theta_start = np.hstack(([d0_dep], par[9:12]))  # [d0, t1, t2, t3]
    limits = np.vstack(([ [0.0,0.0] ], par[12:18].reshape(3,2)))  # rail + RRR
    dt = 0.05

    # Charger deux waypoints
    fname = input("Fichier .xy (exactement 2 waypoints) : ").strip()
    xy = np.loadtxt(fname)
    if xy.shape[0] != 2:
        raise ValueError("Le fichier .xy doit contenir EXACTEMENT 2 points.")

    p1 = xy[0]
    p2 = xy[1]

    # Générer 11 points le long de la ligne
    samples = [p1 + s*(p2 - p1) for s in np.linspace(0,1,11)]

    # Initialisation
    allSM, UpperB, LowerB, MNS, BAS = [], [], [], [], []

    q_current = theta_start.copy()
    q_current = robot.Reach_PRRR(p1, q_current, L, base, dt, limits, a=1, d0_fixed=d0_dep)

    q_avoid = theta_start.copy()
    q_avoid = robot.Reach_PRRR(p1, q_avoid, L, base, dt, limits, a=2, d0_fixed=d0_dep)

    print("Calcul des 11 selfmotions...")

    for k, pf in enumerate(samples):
        print(f" → Selfmotion point {k+1}/11")
        q_current = robot.Reach_PRRR(pf, q_current, L, base, dt, limits, a=1, d0_fixed=d0_dep)
        MNS.append(q_current)

        q_avoid = robot.Reach_PRRR(pf, q_avoid, L, base, dt, limits, a=2, d0_fixed=d0_dep)
        BAS.append(q_avoid)

        # Selfmotion principal (+1)
        SM = SelfMotionPoint(q_avoid, pf, L, base, limits, direction=+1, dt=dt)
        allSM.append(SM)
        UpperB.append(SM[-1])

        # Selfmotion direction -1
        SM = SelfMotionPoint(q_avoid, pf, L, base, limits, direction=-1, dt=dt)
        allSM.append(SM)
        LowerB.append(SM[-1])

    # Transformer listes en np.array
    MNS    = np.vstack(MNS)
    BAS    = np.vstack(BAS)
    UpperB = np.vstack(UpperB)
    LowerB = np.vstack(LowerB)

    # ---- GRAPH 3D ----
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    # Tracé selfmotions (angles rotatifs uniquement t1-t3)
    for i, SM in enumerate(allSM):
        w = 2 if i in [10,11] else 1
        ax.plot(SM[:,1], SM[:,2], SM[:,3], color="black", linewidth=w)

    # UpperBound
    ax.plot(UpperB[:,1], UpperB[:,2], UpperB[:,3], color="Red", linewidth=2, label="Upper Bound")
    UpperP = UpperB[[0,5,10],:]
    ax.plot(UpperP[:,1], UpperP[:,2], UpperP[:,3],"o", color="Red")

    # LowerBound
    ax.plot(LowerB[:,1], LowerB[:,2], LowerB[:,3], color="Red", linewidth=2, label="Lower Bound")
    LowerP = LowerB[[0,5,10],:]
    ax.plot(LowerP[:,1], LowerP[:,2], LowerP[:,3],"o", color="Red")

    # Boundary-Avoid
    ax.plot(BAS[:,1], BAS[:,2], BAS[:,3], color="Blue", linewidth=2, label="Boundary-Avoid. Sol.")
    BASP = BAS[[0,5,10],:]
    ax.plot(BASP[:,1], BASP[:,2], BASP[:,3],"o", color="Blue")

    # Minimum-Norm
    ax.plot(MNS[:,1], MNS[:,2], MNS[:,3], color="Green", linewidth=2, label="Min.-Norm Sol.")
    MNSP = MNS[[0,5,10],:]
    ax.plot(MNSP[:,1], MNSP[:,2], MNSP[:,3],"o", color="Green")

    ax.set_xlabel("θ1")
    ax.set_ylabel("θ2")
    ax.set_zlabel("θ3")
    ax.set_title("Selfmotion le long d'une ligne droite (PRRR)")

    ax.legend()
    plt.tight_layout()
    plt.show()
