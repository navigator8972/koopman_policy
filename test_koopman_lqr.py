import koopman_lqr as klqr
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_solve_lqr():
    #these params do not matter
    ctrl = klqr.KoopmanLQR(k=4, x_dim=4, u_dim=2, x_goal=np.zeros(4), T=100, phi=None, u_affine=None)


    dt = 0.01
    #circular path
    ang = np.arange(0, 1, dt) * 2*np.pi
    T = len(ang)
    ref = np.array([np.cos(ang), np.sin(ang)]).T + np.array([0.0, 1.0])
    ref = np.concatenate((ref, dt*np.array([-np.sin(ang), np.cos(ang)]).T), 1)

    #plot ref
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(ref[:, 0], ref[:, 1], '*')

    #a 2D point mass
    m = 0.01
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]  ])
    B = np.array([  [0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1]  ]) * dt/m
    Q = np.eye(4)
    #less forcusing on tracking velocity
    Q[2,2] = 0.001
    Q[3,3] = 0.001

    R = np.eye(2)*0.01

    #remember to account batch dimension
    K, k = ctrl._solve_lqr(     A=torch.from_numpy(A).unsqueeze(0),
                                B=torch.from_numpy(B).unsqueeze(0),
                                Q=torch.from_numpy(Q).unsqueeze(0),
                                R=torch.from_numpy(R).unsqueeze(0),
                                goals=torch.from_numpy(ref).unsqueeze(0))
    # print(K)
    # print(k)
    
    # apply the control
    x0 = np.random.randn(2) * 0.2
    x0 = x0 + np.array([1.0, 1.0])
    x0 = np.concatenate((x0, np.zeros(2)))

    traj = [x0]
    for i in range(T-1):
        u = -K[i][0].numpy().dot(traj[-1])+k[i][0].numpy()   #batch_size=1
        x_new = A.dot(traj[-1])+B.dot(u)
        traj.append(x_new)
    
    traj = np.array(traj)
    ax.plot(traj[:, 0], traj[:, 1])
    plt.show()
    return


if __name__ == "__main__":
    test_solve_lqr()    