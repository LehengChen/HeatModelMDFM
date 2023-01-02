import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import math
import pathlib
import os
import sys
import argparse
from scipy.io import savemat

t_start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--tauUinv", help="tau_U inv", type=str, default='01')
parser.add_argument("--tauNinv", help="tau_N inv", type=str, default='10')
parser.add_argument("--refT", help="reference T", type=str, default='1') # see readme in datafile
parser.add_argument("--DeltaT", help="Delta T", type=str, default='0')
parser.add_argument("--GK", help="simulate GK model or not", type=int, default=0)
parser.add_argument("--test", help="test case", type=int, default=2) # 1 means discontinuous case, 2 means longer time case
parser.add_argument("--Tnum", help="T num", type=int, default=901)
parser.add_argument("--muu", help="diffusion term of u", type=float, default=0.02)
parser.add_argument("--muq", help="diffusion term of q", type=float, default=0.02)
parser.add_argument("--muQ", help="diffusion term of Q", type=float, default=0.02)
parser.add_argument("--muuGK", help="diffusion term of u(GK)", type=float, default=0.02)
parser.add_argument("--mu_stop", help="set diffusion term to zero", type=int, default=50)
args = parser.parse_args()

torch.set_default_dtype(torch.float64)

sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
torch.set_default_dtype(torch.float64)

# ----------------------------
# Basic parameters
# ----------------------------
pi = math.pi

# Kn number
U = args.tauUinv
N = args.tauNinv
if U == '10':
    U_num = 10
elif U == '1':
    U_num = 1
else:
    U_num = 0.1

if N == '10':
    N_num = 10
elif N == '1':
    N_num = 1
else:
    N_num = 0.1

print(f'tau_U_inv = {U_num}, tau_N_inv = {N_num}')
Kn = N # Not Used
if args.test == 2:
    initial_Kn_name = "TTGdata_T" + args.refT + "_DT" + args.DeltaT + "_init5.mat"
else:
    initial_Kn_name = "TTGdata_T" + args.refT + "_DT" + args.DeltaT + "_init1.mat"

CDFNet_file = "net_params_U" + U + "N" + N + "Coarse1steps1.pkl" # saved model's name
print(f'CDFNet_file_path = {CDFNet_file}')

# prediction group number
prediction_group = [int(args.refT)]
print("prediction group numbers: ", prediction_group)

# ----------------------------
# load the data from Matlab file
# ----------------------------

def read_data(load_fn, T_num, i_num):
    load_data = sio.loadmat(load_fn)
    number = (i_num - 1) * 201 + T_num * 1 - 1 # if T_num > 1 else 0
    u_1 = torch.from_numpy(load_data['U'])[number].to(device)
    q_1 = torch.from_numpy(load_data['q'])[number].to(device)
    Q1D_1 = torch.from_numpy(load_data['QQ'])[number].to(device)

    return u_1, q_1, Q1D_1

def read_data_inintial(load_fn):
    load_data = sio.loadmat(load_fn)

    u_0 = torch.from_numpy(load_data['U0'])[0].to(device).detach()
    q_0 = torch.from_numpy(load_data['q0'])[0].to(device).detach()
    Q1D_0 = torch.from_numpy(load_data['QQ0'])[0].to(device).detach()

    return u_0, q_0, Q1D_0

curr_path = pathlib.Path(__file__).parent.absolute()

parent_path = curr_path.parent

if args.test == 1:
    matlab_path_data = os.path.join(parent_path, "testdata_discontinuous_cases", "U" + U + "N" + N)
elif args.test == 2:
    matlab_path_data = os.path.join(parent_path, "testdata_075_085", "U" + U + "N" + N)

net_data_path = "BTEdataU" + U + "N" + N

from train import load_model
Model = load_model(os.path.join(net_data_path, CDFNet_file)).to(device)

def Flux(u):
    Theta_inv = 1 / u
    return Theta_inv


def CDFmodel(u, q, Q1D, n, T, init_data_file, total_step=200, pos=30):
    L = 2 * math.pi
    dx = L / n
    if args.test == 1:
        Nt = args.Tnum * 100
    else:
        Nt = args.Tnum * 10
    dt = T / Nt
    print(f'CDFmodel prediction-dt:{dt}')
    lam = dt / dx
    dt = T / Nt
    log_every = Nt // total_step
    U_pred_T = []
    U_exact_T = []
    q_pred_T = []
    q_exact_T = []

    muu = args.muu
    muq = args.muq
    muQ = args.muQ

    loss = {'U_L2':[], 'U_L1':[], 'q_L2':[], 'q_L1':[]}

    for i in range(Nt):

        # U_{j+1} and U_{j-1} with periodic boundary
        u_p = torch.cat([u[1:], u[0:1]], dim=0).detach()
        u_m = torch.cat([u[-1:], u[0:-1]], dim=0).detach()
        q_p = torch.cat([q[1:], q[0:1]], dim=0).detach()
        q_m = torch.cat([q[-1:], q[0:-1]], dim=0).detach()
        Q1D_p = torch.cat([Q1D[1:], Q1D[0:1]], dim=0).detach()
        Q1D_m = torch.cat([Q1D[-1:], Q1D[0:-1]], dim=0).detach()

        # F_{j+1} and F_{j-1}
        Theta_inv_m = Flux(u_m)
        Theta_inv = Flux(u)
        Theta_inv_p = Flux(u_p)

        #Schemes
        if args.test == 1:
            u = u - 0.5 * lam * (q_p - q_m) + dt * muu * (u_p + u_m - 2 * u) / dx / dx
        else:
            u = u - 0.5 * lam * (q_p - q_m)

        G0 = Model.G(u.reshape(n, 1), q.reshape(n, 1)).detach() 
        G0 = G0.reshape(1, n)[0]
        M0 = Model.M0(u.reshape(n, 1), q.reshape(n, 1), Q1D.reshape(n, 1)).detach()
        M0 = M0.reshape(1, n)[0]
        M1 = Model.M1(u.reshape(n, 1), q.reshape(n, 1), Q1D.reshape(n, 1)).detach()
        M1 = M1.reshape(1, n)[0]
        exM = Model.exM.detach()
        beta = Model.beta1D.detach()

        if args.test == 1:
            q = q + ( 0.5 * lam * (Theta_inv_p - Theta_inv_m) - exM * 1 / beta * 0.5 * lam * \
                  (Q1D_p-Q1D_m) - dt * M0 * q ) / G0 + dt * muq * (q_p + q_m - 2 * q) / dx / dx
        else:
            q = q + (0.5 * lam * (Theta_inv_p - Theta_inv_m) - exM * 1 / beta * 0.5 * lam * \
                 (Q1D_p - Q1D_m) - dt * M0 * q) / G0
        if args.test == 1:
            Q1D = Q1D - exM * 0.5 * lam *(q_p - q_m) - 1 / beta * dt * M1 * Q1D + dt * muQ * (Q1D_p + Q1D_m - 2 * Q1D) / dx / dx
        else:
            Q1D = Q1D - exM * 0.5 * lam *(q_p - q_m) - 1 / beta * dt * M1 * Q1D


        if i % log_every == log_every - 1:
            print(f"\nCDFmodelPred: i = {i}, T = {dt * i}")
            # In discontinuous case, set the diffusion term to 0 after a few iterations
            if ((i + 1) // log_every) == args.mu_stop:
                muu = 0
                muq = 0
                muQ = 0

            # read exact solution
            u_exact, q_exact, _ = read_data(init_data_file, T_num=(i+1) // log_every, i_num=1)
            u_exact_np = u_exact.detach().cpu().numpy()
            q_exact_np = q_exact.detach().cpu().numpy()
            u_pred_np = u.detach().cpu().numpy()
            q_pred_np = q.detach().cpu().numpy()

            U_pred_T.append(u_pred_np)
            U_exact_T.append(u_exact_np)
            q_pred_T.append(q_pred_np) 
            q_exact_T.append(q_exact_np)

            # L2 and L1 error for CDF model
            Err_rho_L2_u = np.sqrt(sum((u_pred_np - u_exact_np) ** 2) / sum(u_exact_np ** 2))
            Err_rho_L1_u = sum(abs(u_pred_np - u_exact_np)) / sum(abs(u_exact_np))
            Err_rho_L2_q = np.sqrt(sum((q_pred_np - q_exact_np) ** 2) / sum(q_exact_np ** 2))
            Err_rho_L1_q = sum(abs(q_pred_np - q_exact_np)) / sum(abs(q_exact_np))

            loss['U_L2'].append(Err_rho_L2_u)
            loss['U_L1'].append(Err_rho_L1_u)
            loss['q_L2'].append(Err_rho_L2_q)
            loss['q_L1'].append(Err_rho_L1_q)

            print("Pred_u_L2 error:{}".format(Err_rho_L2_u))
            print("Pred_u_L1 error:{}".format(Err_rho_L1_u))
            print("Pred_q_L2 error:{}".format(Err_rho_L2_q))
            print("Pred_q_L1 error:{}".format(Err_rho_L1_q))

            if args.test == 1:
                print(f'mu_u = {muu}, mu_q = {muq}')
    return U_pred_T, U_exact_T, q_pred_T, q_exact_T, loss

def GKModel(u, q, n, T, init_data_file, tauR, tauN, c_v=1, v_g=1, total_step=200, pos=30):
    print(f'tauR = {tauR}, tauN = {tauN}')
    L = 2 * math.pi
    dx = L / n
    Nt = args.Tnum * 100 # some sufficient large number, so dt is small enough
    dt = T / Nt
    lam = dt / dx
    dt = T / Nt
    print(f'GKmodel prediction-dt:{dt}')

    k_bulk = 1 / 3 * c_v * v_g * v_g * tauR
    l_square = 1 / 5 * v_g * v_g * tauR * tauN

    log_every = Nt // total_step
    U_GK_T = []
    q_GK_T = []

    loss = {'U_L2':[], 'U_L1':[], 'q_L2':[], 'q_L1':[]}

    muu = args.muuGK

    for i in range(Nt):

        # U_{j+1} and U_{j-1} with periodic boundary
        u_p = torch.cat([u[1:], u[0:1]], dim=0)
        u_m = torch.cat([u[-1:], u[0:-1]], dim=0)
        q_p = torch.cat([q[1:], q[0:1]], dim=0)
        q_m = torch.cat([q[-1:], q[0:-1]], dim=0)

        if args.test == 1:
            u = u - 1 / c_v * 0.5 * lam * (q_p - q_m) + muu * dt * (u_p + u_m - 2 * u) / dx / dx
        else:
            u = u - 1 / c_v * 0.5 * lam * (q_p - q_m)

        q = q + dt / tauR * ( (-1)*q - k_bulk * 0.5 / dx * (u_p - u_m) + 3 * l_square * (q_p + q_m - 2 * q) / dx / dx )

        if i % log_every == log_every - 1:
            if ((i + 1) // log_every) == args.mu_stop:
                muu = 0
            print(f"\nGK: i = {i}, T = {dt * i}")

            # read exact solution
            u_exact, q_exact, _ = read_data(init_data_file, T_num=(i + 1) // log_every, i_num=1)
            u_exact_np = u_exact.detach().cpu().numpy()
            q_exact_np = q_exact.detach().cpu().numpy()

            u_GK_np = u.detach().cpu().numpy()
            q_GK_np = q.detach().cpu().numpy()

            U_GK_T.append(u_GK_np)  
            q_GK_T.append(q_GK_np) 

            # L2 and L1 error for GK model
            Err_rho_L2_u = np.sqrt(sum((u_GK_np - u_exact_np) ** 2) / sum(u_exact_np ** 2))
            Err_rho_L1_u = sum(abs(u_GK_np - u_exact_np)) / sum(abs(u_exact_np))
            Err_rho_L2_q = np.sqrt(sum((q_GK_np - q_exact_np) ** 2) / sum(q_exact_np ** 2))
            Err_rho_L1_q = sum(abs(q_GK_np - q_exact_np)) / sum(abs(q_exact_np))

            loss['U_L2'].append(Err_rho_L2_u)
            loss['U_L1'].append(Err_rho_L1_u)
            loss['q_L2'].append(Err_rho_L2_q)
            loss['q_L1'].append(Err_rho_L1_q)

            print("GK_u_L2 error:{}".format(Err_rho_L2_u))
            print("GK_u_L1 error:{}".format(Err_rho_L1_u))
            print("GK_q_L2 error:{}".format(Err_rho_L2_q))
            print("GK_q_L1 error:{}".format(Err_rho_L1_q))

    return U_GK_T, q_GK_T, loss


# --------------------------
# Predictions
# --------------------------

Nx = 80
time_step = 2*pi/160

for init_num in prediction_group:
    T_num = args.Tnum
    T = time_step * T_num
    T_NUM = np.arange(T_num)

    init_data_file = os.path.join(matlab_path_data, initial_Kn_name)
    u_0, q_0, Q1D_0 = read_data_inintial(init_data_file)

    # GK solver
    if args.GK == 1:
        U_GK_T, q_GK_T, loss_GK = GKModel(u_0, q_0, Nx, T, init_data_file, tauR=1/U_num, tauN=1/N_num, total_step=T_num)

    # solve CDF equation using CDFNetModel
    U_pred_T, U_exact_T, q_pred_T, q_exact_T, loss_pred = CDFmodel(u_0, q_0, Q1D_0, Nx, T, init_data_file, total_step=T_num)

    # final time
    final_time = format(T, '.1f')
    print("predict group number: ", init_num, ";  final time: ", final_time)


print(f'Time elapsed: {time.time() - t_start}')
