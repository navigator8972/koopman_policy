from enum import Flag
import koopman_policy.koopman_lqr as kpm
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_solve_lqr():
    #these params do not matter
    ctrl = kpm.KoopmanLQR(k=4, x_dim=4, u_dim=2, x_goal=torch.from_numpy(np.zeros(4)), T=100, phi=None, u_affine=None)


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
    K, k, V, v = ctrl._solve_lqr(   A=torch.from_numpy(A).unsqueeze(0),
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

def test_cost_to_go():
    goal = torch.from_numpy(np.zeros(2)).float()
    ctrl = kpm.KoopmanLQR(k=2, x_dim=2, u_dim=2, x_goal=goal, T=15, phi=nn.Identity(), u_affine=None, g_goal=None)
    A = np.eye(2)
    B = np.eye(2)
    Q = np.array([10, 3])

    ctrl._phi_affine = nn.Parameter(torch.from_numpy(A).float())
    ctrl._u_affine = nn.Parameter(torch.from_numpy(B).float())
    ctrl._q_diag_log = nn.Parameter(torch.from_numpy(Q).float().log())

    n_pnts = 50
    x0 = np.array([[[x, y] for y in np.linspace(-10, 10, n_pnts)] for x in np.linspace(-10, 10, n_pnts)])
    x0 = torch.from_numpy(x0).float().view((-1, 2))
    cost_to_go = ctrl.forward_cost_to_go(x0)
    #verify a quadratic cost-to-go for lqr
    img = cost_to_go.detach().numpy().reshape(n_pnts, n_pnts)
    plt.imshow(img)
    plt.show()

import pybullet_envs
import gym
from koopman_policy.utils import KoopmanThinPlateSplineBasis, KoopmanFCNNLift

def test_pendulum():
    torch.manual_seed(0)
    np.random.seed(0)

    env = gym.make('InvertedPendulumBulletEnv-v0')

    env.seed(0)
    env.observation_space.seed(0)
    env.action_space.seed(0)

    #collect data
    n_trajs = 1000
    n_steps = 200

    x_dim = env.observation_space.shape[0]
    u_dim = env.action_space.shape[0]
    n_k = 200

    trajs = np.random.rand(n_trajs, n_steps, x_dim) * 2 -1 
    u = np.random.rand(n_trajs, n_steps, u_dim) * 2 -1 
    for i in range(n_trajs):
        trajs[i, 0, :] = env.reset()
        for j in range(n_steps-1):
            u[i, j] = env.action_space.sample()
            x_new, r, done, info = env.step(u[i, j])
            trajs[i, j+1, :] = x_new
    
    #statistics of data
    mean = np.mean(trajs, axis=(0, 1))
    std = np.std(trajs, axis=(0, 1))
    #whiten data
    # trajs_normed = (np.array(trajs) - mean[None, None, :])/std[None, None, :]
    max = np.float32(np.max(trajs, axis=(0, 1)))
    min = np.float32(np.min(trajs, axis=(0, 1)))
    print('max, min:', max, min)
    print('mean, std;', mean, std)

    phi_fixed_basis = KoopmanThinPlateSplineBasis(in_dim=x_dim, n_basis=n_k-x_dim, center_dist_box_scale=max-min)
    phi_fcnn_basis = KoopmanFCNNLift(in_dim=x_dim, out_dim=n_k, hidden_dim=[4, 4])
    ctrl = kpm.KoopmanLQR(k=n_k, x_dim=x_dim, u_dim=u_dim, x_goal=torch.zeros(5).float(), T=5, phi=phi_fcnn_basis, u_affine=None)
    
    # ctrl.cuda()
    ctrl.fit_koopman(torch.from_numpy(trajs).float(), torch.from_numpy(u).float(), 
        train_phi=True, 
        train_phi_inv=False,
        train_metric=False,
        ls_factor=1,
        recurr = 1,
        n_itrs=10, 
        lr=2.5e-3, 
        verbose=True)
    # ctrl.cpu()

    #for prediction
    # test_step = 300

    # x_0 = env.reset()
    # u = np.array([(-1) ** (np.rint(np.arange(test_step) / 30))]).T[np.newaxis, :, :]
    
    # test_traj = [x_0]
    # pred_traj = [ctrl._phi(torch.from_numpy(x_0).float().unsqueeze(0)).detach().numpy()]    #unsqueeze for the batch dimension
    # for t in range(test_step-1):
    #     x_new, r, done, info = env.step(u[:, t, :])
        
    #     x_pred = ctrl.predict_koopman(torch.from_numpy(pred_traj[-1]).float(), torch.from_numpy(u[:, t, :]).float())
    #     test_traj.append(x_new)
    #     pred_traj.append(x_pred.detach().numpy())

    # test_traj = ctrl._phi(torch.from_numpy(np.array(test_traj)[None, ...]).float()).detach().numpy()
    # pred_traj = np.swapaxes(np.array(pred_traj), 0, 1)
    # pred_traj[:, :, :5]=pred_traj[:, :, :5]*std+mean

    # fig = plt.figure(figsize=(16, 8))
    # ax = fig.add_subplot(121)
    # ax.plot(np.arange(test_traj.shape[1]), u[0, :, 0])
    # ax = fig.add_subplot(122)
    # ax.plot(np.arange(test_traj.shape[1]), test_traj[0, :, 0] , '.b', markersize=1)
    # ax.plot(np.arange(test_traj.shape[1]), test_traj[0, :, 1] , '.g', markersize=1)
    # ax.plot(np.arange(test_traj.shape[1]), test_traj[0, :, 2] , '.r', markersize=1)
    # ax.plot(np.arange(test_traj.shape[1]), test_traj[0, :, 4] , '.y', markersize=1)
    # ax.plot(np.arange(pred_traj.shape[1]), pred_traj[0, :, 0], 'b-')
    # ax.plot(np.arange(pred_traj.shape[1]), pred_traj[0, :, 1], 'g-')
    # ax.plot(np.arange(pred_traj.shape[1]), pred_traj[0, :, 2], 'r-')
    # ax.plot(np.arange(pred_traj.shape[1]), pred_traj[0, :, 4], 'y-')
    # plt.show()



    #for test
    x_0 = env.reset()
    test_traj = [x_0]
    tol_r = 0

    Q = np.ones(n_k) * 1e-3
    #focusing on theta dimension 
    # Q[0] = 0.01
    Q[2] = 1.
    Q[3] = 1.
    # Q[4] = 1e-6

    R = np.ones(u_dim)*1e-4
    #state format is np.array([x, vx, np.cos(self.theta), np.sin(self.theta), theta_dot])
    ctrl._x_goal = nn.Parameter(torch.from_numpy(np.concatenate((x_0[:2], np.array([1, 0, 0])))).float().unsqueeze(0))
    ctrl._q_diag_log = nn.Parameter(torch.from_numpy(Q).float().log())
    ctrl._r_diag_log = nn.Parameter(torch.from_numpy(R).float().log())
    u_lst = []
    for t in range(300):
        u_klqr = ctrl(torch.from_numpy(test_traj[-1]).float().unsqueeze(0)).detach().numpy()[0]
        u_lst.append(u_klqr)
        x_new, r, done, info = env.step(u_klqr) 
        test_traj.append(x_new)
        tol_r += r
        if done:
            print('Terminated at Step {0}'.format(t))
            break

    test_traj = np.array(test_traj)
    u_lst = np.array(u_lst)

    print(tol_r)    
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    ax.plot(np.arange(test_traj.shape[0]-1), u_lst[:, 0])
    ax = fig.add_subplot(122)
    ax.plot(np.arange(test_traj.shape[0]-1), test_traj[:-1, 0], 'b', markersize=1)
    ax.plot(np.arange(test_traj.shape[0]-1), test_traj[:-1, 1], 'g', markersize=1)
    ax.plot(np.arange(test_traj.shape[0]-1), test_traj[:-1, 2], 'r', markersize=1)
    ax.plot(np.arange(test_traj.shape[0]-1), test_traj[:-1, 3], 'c', markersize=1)
    ax.plot(np.arange(test_traj.shape[0]-1), test_traj[:-1, 4], 'y', markersize=1)
    plt.show()

    return

if __name__ == "__main__":
    # test_solve_lqr()
    # test_mpc()    
    #test_cost_to_go()
    test_pendulum()
