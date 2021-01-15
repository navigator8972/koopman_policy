import koopman_policy.koopman_lqr as kpm
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_solve_lqr():
    #these params do not matter
    ctrl = kpm.KoopmanLQR(k=4, x_dim=4, u_dim=2, x_goal=np.zeros(4), T=100, phi=None, u_affine=None)


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

def test_koopman_fit():
    '''
    forced van der pol example from Korda & Mezic 2016, arxiv/1611.03537 
    '''

    def vanderpol2d(x, u, dt=0.01):
        #note the first dim is batch dimension
        x_dot_1 = 2*x[:, 1]
        x_dot_2 = -0.8*x[:, 0] + 2*x[:, 1] - 10*x[:, 0]**2*x[:, 1] + u[:, 0]
        return x + np.array([x_dot_1, x_dot_2]).T*dt
    
    n_trajs = 20
    n_steps = 1000

    u = np.random.rand(n_trajs, n_steps, 1) * 2 - 1
    x_0 = np.random.rand(n_trajs, 2) * 2 - 1

    trajs = [x_0]
    for t in range(n_steps-1):
        x_new = vanderpol2d(trajs[-1], u[:, t, :])
        trajs.append(x_new)
    
    trajs = np.swapaxes(np.array(trajs), 0, 1)
    
    #it seems a linear embedding sometimes gives a slightly better fit even without the regularization terms.
    #is there a way we could perform dual optimization to tune the weight of regularization as well?
    ctrl = kpm.KoopmanLQR(k=2, x_dim=2, u_dim=1, x_goal=torch.zeros(4).float(), T=100, phi=[16, 16], u_affine=None)
    ctrl.cuda()
    ctrl.fit_koopman(torch.from_numpy(trajs).float().cuda(), torch.from_numpy(u).float().cuda(), 
        train_phi=True, 
        train_phi_inv=True,
        train_metric=True,
        ls_factor=1.,
        n_itrs=500, 
        lr=1e-3, 
        verbose=True)
    ctrl.cpu()

    #for test
    x_0 = np.ones((1, 2)) * 0.5
    #sinoidal input, this is different from the doc
    u = np.array([np.sin(2*np.pi*0.3*np.linspace(0, 1, n_steps))]).T[np.newaxis, :, :]
    test_traj = [x_0]
    pred_traj = [ctrl._phi(torch.from_numpy(x_0).float().unsqueeze(0)).detach().numpy()]    #unsqueeze for the batch dimension
    for t in range(n_steps-1):
        x_new = vanderpol2d(test_traj[-1], u[:, t, :])
        
        x_pred = ctrl.predict_koopman(torch.from_numpy(pred_traj[-1]).float(), torch.from_numpy(u[:, t, :]).float())
        # try one step pred?
        # x_pred = ctrl.predict_koopman(torch.from_numpy(test_traj[-1]).float(), torch.from_numpy(u[:, t, :]).float())
        test_traj.append(x_new)
        pred_traj.append(x_pred.detach().numpy())
    
    test_traj = np.swapaxes(np.array(test_traj), 0, 1)
    #note we cannot directly compare to the test traj in the unlifted space...
    test_traj = ctrl._phi(torch.from_numpy(test_traj).float()).detach().numpy()
    pred_traj = np.swapaxes(np.array(pred_traj), 0, 1)
    # print(pred_traj.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_traj.shape[1]), test_traj[0, :, 0] , 'b.')
    ax.plot(np.arange(test_traj.shape[1]), test_traj[0, :, 1] , 'g.')
    ax.plot(np.arange(pred_traj.shape[1]), pred_traj[0, :, 0, 0], 'b-')
    ax.plot(np.arange(pred_traj.shape[1]), pred_traj[0, :, 0, 1], 'g-')
    plt.show()
    return

import torch.nn as nn

def test_mpc():
    goal = torch.from_numpy(np.zeros(4)).float()
    ctrl = kpm.KoopmanLQR(k=4, x_dim=4, u_dim=2, x_goal=goal, T=15, phi=nn.Identity(), u_affine=None, g_goal=None)
    
    dt = 0.01
    #circular path
    ang = np.arange(0, 1, dt) * 2*np.pi
    T = len(ang)
    ref = np.array([np.cos(ang), np.sin(ang)]).T + np.array([0.0, 1.0])
    ref = np.concatenate((ref, dt*np.array([-np.sin(ang), np.cos(ang)]).T), 1)

    #plot ref
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    # ax.plot(ref[:, 0], ref[:, 1], '*')

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

    ctrl._phi_affine = nn.Parameter(torch.from_numpy(A).float())
    ctrl._u_affine = nn.Parameter(torch.from_numpy(B).float())
    ctrl._q_diag_log = nn.Parameter(torch.from_numpy(np.diag(Q)).float().log())
    ctrl._r_diag_log = nn.Parameter(torch.from_numpy(np.diag(R)).float().log())
    

    #for test
    x_0 = np.concatenate((np.ones(2) * 0.5, np.zeros(2)))
    #sinoidal input, this is different from the doc
    test_traj = [x_0]
    for t in range(T-1):
        ctrl._x_goal = nn.Parameter(torch.from_numpy(ref[t]).float())
        u = ctrl(torch.from_numpy(test_traj[-1]).float().unsqueeze(0)).detach().numpy()[0]
        x_new = A.dot(test_traj[-1]) + B.dot(u) + np.concatenate((np.zeros(2), np.random.randn(2)*0.5))
        test_traj.append(x_new)
    
    test_traj = np.array(test_traj)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(ref[:, 0], ref[:, 1], '*')
    ax.plot(test_traj[:, 0], test_traj[:, 1])
    plt.show()

    return

if __name__ == "__main__":
    # test_solve_lqr()
    # test_koopman_fit()
    test_mpc()    