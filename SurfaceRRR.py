#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SurfaceRRR.py

Affiche 11 courbes de selfmotion (theta1,theta2,theta3)
pour 11 points le long d'une ligne droite entre deux waypoints :
    - waypoint 1
    - 9 points intermédiaires
    - waypoint 2
Chaque point produit 2 selfmotions dans les directions +1 et -1,
mais ici on trace seulement le selfmotion principal (direction +1)
pour lisibilité. On peut activer les deux directions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import RobotRRR as robot

# ---- SELFMOTION POUR UN POINT FIXE ----
def SelfMotionPoint(theta0, pf, L, base, limits,
                    direction=+1, dt=0.05,
                    g_null=0.25, g_task=0.15,
                    max_iter=500, eps=1e-7):

    theta = theta0.copy()
    traj = [theta.copy()]
    I = np.eye(4)

    # direction vers limite haute ou basse
    target = limits[:,1] if direction == 1 else limits[:,0]

    for _ in range(max_iter):

        if robot.hit_limit(theta, limits):
            break

        p = robot.forward(theta, L, base)
        J = robot.jacobian(theta, L)
        Jp = np.linalg.pinv(J)

        # maintenir la position
        dp = pf - p
        dtheta_task = g_task * (Jp @ dp)

        # selfmotion = projection sur le noyau
        h = target - theta
        N = I - Jp @ J
        dtheta_null = g_null * (N @ h)

        # arrêt en stagnation
        if np.linalg.norm(dtheta_null) < eps:
            break

        dtheta = dtheta_task + dtheta_null
        theta = robot.update_with_limits(theta, dtheta, dt, limits)

        traj.append(theta.copy())

    return np.array(traj)

# ---- MAIN SCRIPT ----
if __name__ == "__main__":
    print("SurfaceRRR: Version 2.0")
    
    # Charger RobotRRR.par
    par = np.loadtxt("RobotRRR.par")    
    L = par[0:3] # L1 L2 L3
    base = par[7:9]
    theta_start = par[9:13].reshape(4) # d, q1, q2, q3
    limits = par[13:21].reshape(4,2) # limits for d, q1, q2, q3
    dt = 0.05 	# par[22] / 1000

    # Charger deux waypoints
    fname = input("Fichier .xy (exactement 2 waypoints) : ").strip()
    xy = np.loadtxt(fname)
    if xy.shape[0] != 2:
        raise ValueError("Le fichier .xy doit contenir EXACTEMENT 2 points.")

    p1 = xy[0]
    p2 = xy[1]

    # Générer 11 points le long de la ligne
    samples = [p1 + s*(p2 - p1) for s in np.linspace(0,1,11)]

    # Calculer les 11 selfmotions vers upper et lower limits
    allSM = []
    UpperB, LowerB, MNS, BAS = [], [], [], []
    theta_current = theta_start.copy()
    theta_current = robot.Reach(p1,theta_current,L,base,dt,limits,a=1)
    
    theta_avoid = theta_start.copy()
    theta_avoid = robot.Reach(p1,theta_avoid,L,base,dt,limits,a=2)
    
    print("Calcul des 11 selfmotions...")
    for k, pf in enumerate(samples):
        print(f" → Selfmotion point {k+1}/11")
        theta_current = robot.Reach(pf,theta_current,L,base,dt,limits,a=1)
        MNS.append(theta_current)

        theta_avoid = robot.Reach(pf,theta_avoid,L,base,dt,limits,a=2)
        BAS.append(theta_avoid)

        # On utilise le theta courant, soit le minmum-norn solution (BAS)
        SM = SelfMotionPoint(theta_avoid, pf, L, base, limits, direction=+1, dt=dt)
        allSM.append(SM)
        UpperB.append(SM[-1])
        
        SM = SelfMotionPoint(theta_avoid, pf, L, base, limits, direction=-1, dt=dt)
        allSM.append(SM)
        LowerB.append(SM[-1])
        
        # print(k,UpperB[k],BAS[k],MNS[k],LowerB[k])

    # Transformation de chaque liste de np.array en un seul np.array    
    MNS    = np.vstack(MNS)     # MNS: Minimum-Norm Solution
    BAS    = np.vstack(BAS)     # BAS: Boundary-Avoidance Solution
    UpperB = np.vstack(UpperB)  # UpperB: Upper-Bound Solution
    LowerB = np.vstack(LowerB)  # LowerB: Lower-Bound Solution
     
    # ---- GRAPH 3D ----
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    for i, SM in enumerate(allSM):
        if i == 10 or i == 11: 
            w = 2 
        else: 
            w = 1
        ax.plot(SM[:,0], SM[:,1], SM[:,2],
                color="black",
                linewidth=w)
    
    ax.plot(UpperB[:,0], UpperB[:,1], UpperB[:,2],
        color="Red",
        linewidth=2,
        label="Upper Bound")
    UpperP = UpperB[[0,5,10],:]
    ax.plot(UpperP[:,0], UpperP[:,1], UpperP[:,2],"o", color="Red")
  
    ax.plot(LowerB[:,0], LowerB[:,1], LowerB[:,2],
        color="Red",
        linewidth=2,
        label="Lower Bound")
    LowerP = LowerB[[0,5,10],:]
    ax.plot(LowerP[:,0], LowerP[:,1], LowerP[:,2],"o", color="Red")

    ax.plot(BAS[:,0], BAS[:,1], BAS[:,2],
        color="Blue",
        linewidth=2,
        label="Boundary-Avoid. Sol.")
    BASP = BAS[[0,5,10],:]
    ax.plot(BASP[:,0], BASP[:,1], BASP[:,2],"o", color="Blue")
    
    ax.plot(MNS[:,0], MNS[:,1], MNS[:,2],
        color="Green",
        linewidth=2,
        label="Min.-Norm Sol.")
    MNSP = MNS[[0,5,10],:]
    ax.plot(MNSP[:,0], MNSP[:,1], MNSP[:,2],"o", color="Green")
    
    ax.set_xlabel("θ1")
    ax.set_ylabel("θ2")
    ax.set_zlabel("θ3")
    ax.set_title("Selfmotion le long d'une ligne droite")

    ax.legend()
    plt.tight_layout()
    plt.show()

