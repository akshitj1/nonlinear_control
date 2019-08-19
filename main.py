import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

c_x = c_x_theta = c_y = c_z = c_theta = 1.0
t_final = 2
num_space_segs = 100
theta_err = np.deg2rad(45)
phi_err = np.deg2rad(45)
q_err = R.from_euler('ZX', [phi_err, theta_err]).as_quat()


def q_to_theta(q):
    # theta will always lie in [0,180]
    theta = R.from_quat(q).as_euler('ZXZ')[1]
    return theta

def get_ql(y):
    q = y[:4]
    l = y[4:]
    q = q/np.linalg.norm(q)
    return q,l

def optimum_angular_velocities(q, l):
    theta = q_to_theta(q)

    w_opt_lambdas = np.array([
        [-l[1], l[0], l[3], -l[2]],
        [-l[2], -l[3], l[0], l[1]],
        [-l[3], l[2], -l[1], l[0]]
    ])

    w_opt_den = 1 / 4 * np.array([
        1 / (c_x + c_x_theta * theta ** 2),
        1 / c_y,
        1 / c_z])

    w_opt = w_opt_den * np.matmul(w_opt_lambdas, q.T)
    return w_opt



def singe_frame_ode(y):
    assert len(y.shape) == 1
    EPS = 1e-6
    q, l = get_ql(y)
    w_opt = optimum_angular_velocities(q, l)

    # in body frame
    q_dot_W = np.array([
        [-q[1], q[0], q[3], -q[2]],
        [-q[2], -q[3], q[0], q[1]],
        [-q[3], q[2], -q[1], q[0]]
    ])

    q_dot = 1 / 2 * np.matmul(q_dot_W.T, w_opt)

    l_dot_lambda = np.array([
        [q[0], -l[1], -l[2], -l[3]],
        [-q[1], l[0], -l[3], l[2]],
        [-q[2], l[3], l[0], -l[1]],
        [q[3], -l[2], l[1], l[0]]
    ])

    theta = q_to_theta(q)
    # if theta<EPS: 1 / (1 - theta**2 / 6)
    k = (8 * (c_theta + c_x_theta * w_opt[0] ** 2) * theta) / np.sin(theta) # (1 - (q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2)**2)**0.5
    l_dot = 1 / 2 * np.matmul(l_dot_lambda, np.concatenate(([k],  w_opt)))

    y_dot = np.concatenate((q_dot, l_dot))
    assert not np.isnan(y_dot).any()
    return y_dot


def ode_fn(x, y):
    return np.apply_along_axis(func1d=singe_frame_ode, axis=0, arr=y)


def bc_fn(ya, yb):
    q_t0,_ = get_ql(ya)
    q_res = q_t0 - q_err
    _, l_res = get_ql(yb)
    y_res = np.concatenate((q_res, l_res))
    # print(y_res)
    assert np.all(np.abs(y_res) < 10)
    return y_res


def to_angular_coordinate_frame(y):
    q, l = get_ql(y)
    w_opt = optimum_angular_velocities(q,l)
    euler_angles = R.from_quat(q).as_euler('ZXZ', degrees=True)
    phi, theta = euler_angles[0], euler_angles[1]
    return np.concatenate(([phi, theta], w_opt))

def eval(res):
    t_range = np.linspace(0, t_final, num=100)
    ql_sol = res.sol(t_range)
    ngl_sol = np.apply_along_axis(to_angular_coordinate_frame, 0, ql_sol)
    phi = ngl_sol[0]
    theta = ngl_sol[1]
    w_opt = ngl_sol[2:]
    return theta, phi, w_opt

def plot(theta, w_opt):
    axes = ['x', 'y', 'z']
    plt.figure(figsize=(7,10))
    for i in range(3):
        plt.subplot(3,1,i+1)
        y_plot = w_opt[i]
        plt.plot(theta, y_plot, label='y_'+axes[i])
        plt.legend()
        plt.xlabel("theta")
        plt.ylabel("omega_{}".format(axes[i]))
    plt.show()


if __name__ == '__main__':
    x_time_space = np.linspace(0, t_final, num=num_space_segs)

    # let's linerarly interpolate from pitch_inital, twist_initial to 0
    theta_space = np.linspace(theta_err, 0, num=num_space_segs, endpoint=False)
    phi_space = np.linspace(phi_err, 0, num=num_space_segs, endpoint=False)
    phi_theta_space = np.vstack((phi_space, theta_space)).T
    to_quat = lambda phi_theta: R.from_euler('ZX', phi_theta).as_quat()
    q_guess = np.apply_along_axis(to_quat, 1, phi_theta_space).T

    lambda_guess = np.linspace(1.0, 0, num=num_space_segs) * np.ones((4, num_space_segs))
    y_guess = np.vstack((q_guess, lambda_guess))
    res = solve_bvp(ode_fn, bc_fn, x_time_space, y_guess, verbose=2, tol=1e-3, max_nodes=500)
    print("BVP finished successfully!!!")

    theta, phi, w_opt = eval(res)
    plot(theta, w_opt)
    print('eval finished')
