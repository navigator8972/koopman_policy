from enum import Flag
import koopman_policy.koopman_lqr as kpm
import torch
import numpy as np
import matplotlib.pyplot as plt
from koopman_policy.utils import KoopmanThinPlateSplineBasis, KoopmanFCNNLift

def test_koopman_fit():
    '''
    forced van der pol example from Korda & Mezic 2016, arxiv/1611.03537 
    '''
    torch.manual_seed(0)
    np.random.seed(0)

    def vanderpol2d_ode(x, u):
        #note the first dim is batch dimension
        x_dot_1 = 2*x[:, 1]
        x_dot_2 = -0.8*x[:, 0] + 2*x[:, 1] - 10*x[:, 0]**2*x[:, 1] - u[:, 0]
        return np.array([x_dot_1, x_dot_2]).T
    
    def vanderpol2d(x, u, dt=0.01):
        #use runge-kutta-4order to integrate ode
        k1 = vanderpol2d_ode(x, u)
        k2 = vanderpol2d_ode(x+dt/2*k1, u)
        k3 = vanderpol2d_ode(x+dt/2*k2, u)
        k4 = vanderpol2d_ode(x+dt*k1, u)
        return x + (k1+2*k2+2*k3+k4)*dt/6

    n_trajs = 1000
    n_steps = 200

    u = np.random.rand(n_trajs, n_steps, 1) * 2 -1 
    x_0 = np.random.rand(n_trajs, 2) * 1 - 0.5

    trajs = [x_0]
    for t in range(n_steps-1):
        x_new = vanderpol2d(trajs[-1], u[:, t, :])
        trajs.append(x_new)
    
    trajs = np.swapaxes(np.array(trajs), 0, 1)

    x_dim=2
    u_dim=1
    n_k = 100

    phi_fixed_basis = KoopmanThinPlateSplineBasis(in_dim=x_dim, n_basis=n_k-x_dim, center_dist_box_scale=1.)
    phi_fcnn_basis = KoopmanFCNNLift(in_dim=x_dim, out_dim=n_k, hidden_dim=[4, 4])
    #it seems a linear embedding sometimes gives a slightly better fit even without the regularization terms.
    #is there a way we could perform dual optimization to tune the weight of regularization as well?
    #ctrl = kpm.KoopmanLQR(k=20, x_dim=x_dim, u_dim=u_dim, x_goal=torch.zeros(4).float(), T=100, phi=[32, 32], u_affine=None)
    ctrl = kpm.KoopmanLQR(k=n_k, x_dim=x_dim, u_dim=u_dim, x_goal=torch.zeros(4).float(), T=100, phi=phi_fcnn_basis, u_affine=None)
    # ctrl = kpm.KoopmanLQR(k=n_k, x_dim=x_dim, u_dim=u_dim, x_goal=torch.zeros(4).float(), T=100, phi=phi_fixed_basis, u_affine=None)

    ctrl.cuda()
    ctrl.fit_koopman(torch.from_numpy(trajs).float().cuda(), torch.from_numpy(u).float().cuda(), 
        train_phi=True, 
        train_phi_inv=False,
        train_metric=False,
        ls_factor=1,
        recurr = 1,
        n_itrs=10, 
        lr=1e-1, 
        verbose=True)
    ctrl.cpu()


    #for test
    x_0 = np.ones((1, 2)) * 0.5
    #sinoidal input, this is different from the doc
    # u = np.array([np.sin(2*np.pi*0.3*np.linspace(0, 1, n_steps))]).T[np.newaxis, :, :]
    #bang-bang input, same as the doc
    test_step = 300


    u = np.array([(-1) ** (np.rint(np.arange(test_step) / 30))]).T[np.newaxis, :, :]
    
    test_traj = [x_0]
    pred_traj = [ctrl._phi(torch.from_numpy(x_0).float().unsqueeze(0)).detach().numpy()]    #unsqueeze for the batch dimension
    for t in range(test_step-1):
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

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    ax.plot(np.arange(test_traj.shape[1]), u[0, :, 0])
    ax = fig.add_subplot(122)
    ax.plot(np.arange(test_traj.shape[1]), test_traj[0, :, 0] , '.b', markersize=1)
    ax.plot(np.arange(test_traj.shape[1]), test_traj[0, :, 1] , '.g', markersize=1)
    ax.plot(np.arange(pred_traj.shape[1]), pred_traj[0, :, 0, 0], 'b-')
    ax.plot(np.arange(pred_traj.shape[1]), pred_traj[0, :, 0, 1], 'g-')
    plt.show()
    return


if __name__ == "__main__":
    test_koopman_fit()
