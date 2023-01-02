import scipy.io as sio
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import time
import pathlib
import os, glob
from collections import namedtuple
import argparse
from torch.optim import lr_scheduler

# ------------------------Model---------------------------------------
class CDFNet(nn.Module):
    def __init__(self, num1, num2, initial_beta1D=10):
        super(CDFNet, self).__init__()
        
        self.w0 = nn.Parameter(torch.FloatTensor([1])) # learnable activation function slope
        self.beta1D= nn.Parameter(torch.FloatTensor([initial_beta1D]))
        self.exM= nn.Parameter(torch.FloatTensor([0.33])) # exM means gamma in the paper

        self.l0_G = torch.nn.Linear(2, num1)
        self.l1_G = torch.nn.Linear(num1, num1)
        self.l2_G = torch.nn.Linear(num1, num1)
        self.l3_G = torch.nn.Linear(num1, num1)
        self.l4_G = torch.nn.Linear(num1, 1)

        self.l0_M0 = torch.nn.Linear(3, num2)
        self.l1_M0 = torch.nn.Linear(num2, num2)
        self.l2_M0 = torch.nn.Linear(num2, num2)
        self.l3_M0 = torch.nn.Linear(num2, num2)
        self.l4_M0 = torch.nn.Linear(num2, 1)

        self.l0_M1 = torch.nn.Linear(3, num2)
        self.l1_M1 = torch.nn.Linear(num2, num2)
        self.l2_M1 = torch.nn.Linear(num2, num2)
        self.l3_M1 = torch.nn.Linear(num2, num2)
        self.l4_M1 = torch.nn.Linear(num2, 1)


    def G(self, u, q):
        var_in = torch.cat((u, torch.zeros_like(q)), dim=1) # values of q are not used
        z1_G = torch.sin(self.l0_G(var_in) * self.w0)
        z2_G = torch.sin(self.l1_G(z1_G) * self.w0)
        z3_G = torch.sin(self.l2_G(z2_G) * self.w0)
        z4_G = torch.sin(self.l3_G(z3_G) * self.w0)
        z5_G = F.softplus(self.l4_G(z4_G) * self.w0)
        return z5_G

    def M0(self, u, q, Q1D):
        var_in = torch.cat((u, q, Q1D), dim=1)
        z1_M0 = torch.sin(self.l0_M0(var_in) * self.w0)
        z2_M0 = torch.sin(self.l1_M0(z1_M0) * self.w0)
        z3_M0 = torch.sin(self.l2_M0(z2_M0) * self.w0)
        z4_M0 = torch.sin(self.l3_M0(z3_M0) * self.w0)
        z5_M0 = F.softplus(self.l4_M0(z4_M0) * self.w0)
        return z5_M0

    def M1(self, u, q, Q1D):
        var_in = torch.cat((u, q, Q1D), dim=1)
        z1_M1 = torch.sin(self.l0_M1(var_in) * self.w0)
        z2_M1 = torch.sin(self.l1_M1(z1_M1) * self.w0)
        z3_M1 = torch.sin(self.l2_M1(z2_M1) * self.w0)
        z4_M1 = torch.sin(self.l3_M1(z3_M1) * self.w0)
        z5_M1 = F.softplus(self.l4_M1(z4_M1) * self.w0)
        return z5_M1

    def forward(self, u, q, Q1D, lam, dt, K, n):
        '''K means batchsize'''
        
        dt = dt / Inner_step
        lam = lam / Inner_step
        u_new = u
        q_new = q
        Q1D_new = Q1D
        for i in range(Inner_step):
            # U_{j+1} and U_{j-1}
            # periodic boundary
            u_p = torch.cat([u[:, 1:], u[:, 0:1]], dim=1)
            u_m = torch.cat([u[:, -1:], u[:, 0:-1]], dim=1)

            Q1D_p = torch.cat([Q1D[:, 1:], Q1D[:, 0:1]], dim=1)
            Q1D_m = torch.cat([Q1D[:, -1:], Q1D[:, 0:-1]], dim=1)

            q_p = torch.cat([q[:, 1:], q[:, 0:1]], dim=1)
            q_m = torch.cat([q[:, -1:], q[:, 0:-1]], dim=1)

            Theta_inv = 1 / u
            Theta_inv_p = 1 / u_p
            Theta_inv_m = 1 / u_m

            G_c = self.G(u.reshape(K * n, 1), q.reshape(K * n, 1))
            G_c = G_c.reshape(K, n) 

            M0_c = self.M0(u.reshape(K * n, 1), q.reshape(K * n, 1), Q1D.reshape(K * n, 1))
            M0_c = M0_c.reshape(K, n)

            M1_c = self.M1(u.reshape(K * n, 1), q.reshape(K * n, 1), Q1D.reshape(K * n, 1))
            M1_c = M1_c.reshape(K, n)

            # update u, q, Q
            u_new = u - 0.5 * lam * (q_p - q_m)
            q_new = q + ( 0.5 * lam * (Theta_inv_p - Theta_inv_m) - self.exM * 1 / self.beta1D * 0.5 * lam * \
                (Q1D_p-Q1D_m) - dt * M0_c * q ) / G_c
            Q1D_new = Q1D - self.exM * 0.5 * lam *(q_p - q_m) - 1 / self.beta1D * dt * M1_c * Q1D 
            
            u = u_new
            q = q_new
            Q1D = Q1D_new

        return q, Q1D, u

def read_data_multi_step(load_fn, device, step=4, coarse=1):
    load_data = sio.loadmat(load_fn)
    num_init = load_data['num_init'][0, 0]
    QQ1 = load_data['q0']
    Q=np.array(QQ1,dtype=np.float64)
    UU1 = load_data['U0']
    U = np.array(UU1, dtype=np.float64)
    QQ2 = load_data['q']
    Q_plus=np.array(QQ2,dtype=np.float64)

    QQ1D1 = load_data['QQ0']
    Q1DQ=np.array(QQ1D1,dtype=np.float64)
    QQ1D2 = load_data['QQ']
    Q1DQ_plus=np.array(QQ1D2,dtype=np.float64)

    num_data, _ = Q.shape
    num_data_each_init = int(num_data / num_init)
    q0 = []
    q1 = []
    q2 = []
    u0 = []
    u1 = []

    q3 = []
    q4 = []
    q5 = []
    u2 = []
    u3 = []
    u4 = []

    Q1D0 = []
    Q1D1 = []
    Q1D2 = []
    Q1D3 = []
    Q1D4 = []
    Q1D5 = []

    S = int((num_data_each_init * 600 / 900 - 1)/ coarse) # Use only [0, 600dt] data

    for i in range(num_init):
        for j in range(S - step):
            q0.append(Q[i * num_data_each_init + j*coarse])
            q1.append(Q[i * num_data_each_init + j*coarse + 1*coarse])
            q2.append(Q[i * num_data_each_init + j*coarse + 2*coarse])
            q3.append(Q[i * num_data_each_init + j*coarse + 3*coarse])    
            q4.append(Q[i * num_data_each_init + j*coarse + 4*coarse])   
            q5.append(Q[i * num_data_each_init + j*coarse + 5*coarse])  
            Q1D0.append(Q1DQ[i * num_data_each_init + j*coarse])
            Q1D1.append(Q1DQ[i * num_data_each_init + j*coarse + 1*coarse])
            Q1D2.append(Q1DQ[i * num_data_each_init + j*coarse + 2*coarse])
            Q1D3.append(Q1DQ[i * num_data_each_init + j*coarse + 3*coarse])      
            Q1D4.append(Q1DQ[i * num_data_each_init + j*coarse + 4*coarse])     
            Q1D5.append(Q1DQ[i * num_data_each_init + j*coarse + 5*coarse])  
            u0.append(U[i * num_data_each_init + j*coarse])
            u1.append(U[i * num_data_each_init + j*coarse + 1*coarse])
            u2.append(U[i * num_data_each_init + j*coarse + 2*coarse])      
            u3.append(U[i * num_data_each_init + j*coarse + 3*coarse])     
            u4.append(U[i * num_data_each_init + j*coarse + 4*coarse])    

    u_0 = torch.from_numpy(np.array(u0)).to(device)
    q_0 = torch.from_numpy(np.array(q0)).to(device)
    Q1D_0 = torch.from_numpy(np.array(Q1D0)).to(device)

    u_1 = torch.from_numpy(np.array(u1)).to(device)  
    q_1 = torch.from_numpy(np.array(q1)).to(device)
    Q1D_1 = torch.from_numpy(np.array(Q1D1)).to(device)

    q_2 = torch.from_numpy(np.array(q2)).to(device)
    Q1D_2 = torch.from_numpy(np.array(Q1D2)).to(device)

    u_2 = torch.from_numpy(np.array(u2)).to(device)
    q_3 = torch.from_numpy(np.array(q3)).to(device)
    Q1D_3 = torch.from_numpy(np.array(Q1D3)).to(device)

    u_3 = torch.from_numpy(np.array(u3)).to(device)
    q_4 = torch.from_numpy(np.array(q4)).to(device)
    Q1D_4 = torch.from_numpy(np.array(Q1D4)).to(device)

    u_4 = torch.from_numpy(np.array(u4)).to(device)
    q_5 = torch.from_numpy(np.array(q5)).to(device)
    Q1D_5 = torch.from_numpy(np.array(Q1D5)).to(device)

    return u_0, q_0, Q1D_0, u_1, q_1, Q1D_1, q_2, Q1D_2, u_2, q_3, Q1D_3, u_3, q_4, Q1D_4, u_4, q_5, Q1D_5


def load_model(load_model_name):
    model = CDFNet(num1=50,num2=50)
    model.load_state_dict(torch.load(load_model_name))

    return model


def get_file_with_extension(file_path, extension):
    return [f for f in os.listdir(file_path) if f.endswith(extension)]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tauUinv", help="inverse of tauU number", type=str, default='01')
    parser.add_argument("--tauNinv", help="inverse of tauN number", type=str, default='10')
    parser.add_argument("--lambda2", help="coefficient lambda2", type=float, default=0.1)
    parser.add_argument("--lambda3", help="coefficient lambda3", type=float, default=0)
    parser.add_argument("--coarse", help="coarse number ('coarse=10' means that we use 10dt timestep data for training)", type=int, default=1)
    parser.add_argument("--inner_step", help="inner step number", type=int, default=1)
    parser.add_argument("--lr", help="learning rate", type=float, default=100)
    parser.add_argument("--warm", help="warm up number", type=int, default=5)
    parser.add_argument("--num_iters", help="total iteration number", type=int, default=100)
    parser.add_argument("--width", help="layer size", type=int, default=50)
    parser.add_argument("--print_every", help="print results every print_every epochs", type=int, default=1)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    Inner_step = args.inner_step


    # training group number
    train_group = [0, 1, 2, 3, 4]
    print("train group numbers: ", train_group)

    # training Kn number
    tauUinv = args.tauUinv
    tauNinv = args.tauNinv
    print(f"train Kn numbers: tau_U_inv = {tauUinv}, tau_N_inv = {tauNinv}")

    # current path
    curr_path = pathlib.Path(__file__).parent.absolute()

    # parent path
    parent_path = curr_path.parent

    # net path
    net_data_path = os.path.join(curr_path, "BTEdataU" + tauUinv + "N" + tauNinv)
    if not os.path.exists(net_data_path):
        os.makedirs(net_data_path)

    # -------------Generate data-------------
    pi = math.pi 
    n = 80 
    L = 2 * pi 
    h = L / n 
    x = np.linspace(-L / 2, L / 2 - h / 2, n) 
    #T = 23.6778
    dt = 0.5 * h * args.coarse
    lam = dt / h
    print(f'dt = {dt}, dx = {h}')

    # -------------Read data-------------
    DataStruct = namedtuple("matlab_data", "u_0, q_0, Q1D_0, u_1, q_1, Q1D_1, q_2, Q1D_2, u_2, q_3, Q1D_3, u_3, q_4, Q1D_4, u_4, q_5, Q1D_5")

    training_data = []
    matlab_data_path = os.path.join(parent_path,"Data", "U" + tauUinv + "N" + tauNinv)

    for train_group_number in train_group:
        # file name of training group number
        file_path = os.path.join(matlab_data_path, "{}st_init".format(train_group_number))  # 文件路径 *st_init
        matlab_data_file_names = get_file_with_extension(file_path, ".mat")
        for f in matlab_data_file_names:
            u_0, q_0, Q1D_0, u_1, q_1, Q1D_1, q_2, Q1D_2, u_2, q_3, Q1D_3, u_3, q_4, Q1D_4, u_4, q_5, Q1D_5 = read_data_multi_step(os.path.join(file_path, f), device=device, coarse=args.coarse)
            data = DataStruct(u_0, q_0, Q1D_0, u_1, q_1, Q1D_1, q_2, Q1D_2, u_2, q_3, Q1D_3, u_3, q_4, Q1D_4, u_4, q_5, Q1D_5)

            training_data.append(data)

    print(f"training_data len = {len(training_data)}")

    u_0 = training_data[0].u_0
    q_0 = training_data[0].q_0
    Q1D_0 = training_data[0].Q1D_0
    u_1 = training_data[0].u_1
    q_1 = training_data[0].q_1
    Q1D_1 = training_data[0].Q1D_1
    q_2 = training_data[0].q_2
    Q1D_2 = training_data[0].Q1D_2
    u_2 = training_data[0].u_2
    q_3 = training_data[0].q_3
    Q1D_3 = training_data[0].Q1D_3
    u_3 = training_data[0].u_3
    q_4 = training_data[0].q_4
    Q1D_4 = training_data[0].Q1D_4
    u_4 = training_data[0].u_4
    q_5 = training_data[0].q_5
    Q1D_5 = training_data[0].Q1D_5
    for i in range(len(training_data) - 1):
        u_0 = torch.cat((u_0, training_data[i + 1].u_0))
        q_0 = torch.cat((q_0, training_data[i + 1].q_0))
        Q1D_0 = torch.cat((Q1D_0, training_data[i + 1].Q1D_0))
        u_1 = torch.cat((u_1, training_data[i + 1].u_1))
        q_1 = torch.cat((q_1, training_data[i + 1].q_1))
        Q1D_1 = torch.cat((Q1D_1, training_data[i + 1].Q1D_1))
        q_2 = torch.cat((q_2, training_data[i + 1].q_2))
        Q1D_2 = torch.cat((Q1D_2, training_data[i + 1].Q1D_2))
        u_2 = torch.cat((u_2, training_data[i + 1].u_2))
        q_3 = torch.cat((q_3, training_data[i + 1].q_3))
        Q1D_3 = torch.cat((Q1D_3, training_data[i + 1].Q1D_3))
        u_3 = torch.cat((u_3, training_data[i + 1].u_3))
        q_4 = torch.cat((q_4, training_data[i + 1].q_4))
        Q1D_4 = torch.cat((Q1D_4, training_data[i + 1].Q1D_4))
        u_4 = torch.cat((u_4, training_data[i + 1].u_4))
        q_5 = torch.cat((q_5, training_data[i + 1].q_5))
        Q1D_5 = torch.cat((Q1D_5, training_data[i + 1].Q1D_5))

    # ----------------------Train---------------------------------

    model = CDFNet(num1=args.width, num2=args.width).to(device)
    curr_path = pathlib.Path(__file__).parent.absolute()

    # if use pre-trained model
    # model = load_model(...) 
    # model = model.to(device)
    model.train()

    total_size = u_0.size()[0]  
    print(f'u_0.size = {u_0.size()}') 
    parbatch = 100
    batch_size = int(total_size / parbatch)  

    L = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) 


    warm_up = args.warm  # epoch number to warm up(one step Loss)
    num_iter = args.num_iters
    loss_history = np.zeros(num_iter)
    lambda2 = args.lambda2
    lambda3 = args.lambda3
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=0)
    start = time.time()

    for epoch in range(num_iter):

        permutation = torch.randperm(total_size)
        counter = 0
        Loss = 0
        for i in range(0, total_size, batch_size):

            optimizer.zero_grad()

            indices = permutation[i: i + batch_size]
            Batch = len(indices)

            batch_u_0, batch_q_0, batch_Q1D_0, batch_q_1, batch_Q1D_1 = u_0[indices], q_0[indices], Q1D_0[indices], q_1[indices], Q1D_1[indices]

            batch_u_1, batch_q_2, batch_Q1D_2 = u_1[indices], q_2[indices], Q1D_2[indices]
            batch_u_2, batch_q_3, batch_Q1D_3 = u_2[indices], q_3[indices], Q1D_3[indices]
            batch_u_3, batch_q_4, batch_Q1D_4 = u_3[indices], q_4[indices], Q1D_4[indices]
            batch_u_4, batch_q_5, batch_Q1D_5 = u_4[indices], q_5[indices], Q1D_5[indices]

            batch_qpred_1, batch_Q1Dpred_1, batch_upred_1 = model(batch_u_0, batch_q_0, batch_Q1D_0, lam, dt, Batch, n)

            # Gradually warm up
            if epoch < 1 * warm_up:  
                loss = L(batch_qpred_1, batch_q_1) + lambda2 * L(batch_Q1Dpred_1, batch_Q1D_1) 
            elif epoch < 2 * warm_up:
                batch_qpred_2, batch_Q1Dpred_2, batch_upred_2  = model(batch_upred_1, batch_qpred_1, batch_Q1Dpred_1, lam, dt, Batch, n)
                loss = L(batch_qpred_1, batch_q_1) + lambda2 * L(batch_Q1Dpred_1, batch_Q1D_1) + \
                       L(batch_qpred_2, batch_q_2) + lambda2 * L(batch_Q1Dpred_2, batch_Q1D_2) + \
                       lambda3 * L(batch_upred_2, batch_u_2) 
            elif epoch < 3 * warm_up:
                batch_qpred_2, batch_Q1Dpred_2, batch_upred_2  = model(batch_upred_1, batch_qpred_1, batch_Q1Dpred_1, lam, dt, Batch, n)
                batch_qpred_3, batch_Q1Dpred_3, batch_upred_3  = model(batch_upred_2, batch_qpred_2, batch_Q1Dpred_2, lam, dt, Batch, n)
                loss = L(batch_qpred_1, batch_q_1) + lambda2 * L(batch_Q1Dpred_1, batch_Q1D_1) + \
                       L(batch_qpred_2, batch_q_2) + lambda2 * L(batch_Q1Dpred_2, batch_Q1D_2) + \
                       L(batch_qpred_3, batch_q_3) + lambda2 * L(batch_Q1Dpred_3, batch_Q1D_3) + \
                       lambda3 * (L(batch_upred_2, batch_u_2) + L(batch_upred_3, batch_u_3))
            elif epoch < 4 * warm_up:
                batch_qpred_2, batch_Q1Dpred_2, batch_upred_2  = model(batch_upred_1, batch_qpred_1, batch_Q1Dpred_1, lam, dt, Batch, n)
                batch_qpred_3, batch_Q1Dpred_3, batch_upred_3  = model(batch_upred_2, batch_qpred_2, batch_Q1Dpred_2, lam, dt, Batch, n)
                batch_qpred_4, batch_Q1Dpred_4, batch_upred_4  = model(batch_upred_3, batch_qpred_3, batch_Q1Dpred_3, lam, dt, Batch, n)
                loss = L(batch_qpred_1, batch_q_1) + lambda2 * L(batch_Q1Dpred_1, batch_Q1D_1) + \
                       L(batch_qpred_2, batch_q_2) + lambda2 * L(batch_Q1Dpred_2, batch_Q1D_2) + \
                       L(batch_qpred_3, batch_q_3) + lambda2 * L(batch_Q1Dpred_3, batch_Q1D_3) + \
                       L(batch_qpred_4, batch_q_4) + lambda2 * L(batch_Q1Dpred_4, batch_Q1D_4) + \
                       lambda3 * (L(batch_upred_2, batch_u_2) + L(batch_upred_3, batch_u_3) + L(batch_upred_4, batch_u_4))
            else:
                batch_qpred_2, batch_Q1Dpred_2, batch_upred_2  = model(batch_upred_1, batch_qpred_1, batch_Q1Dpred_1, lam, dt, Batch, n)
                batch_qpred_3, batch_Q1Dpred_3, batch_upred_3  = model(batch_upred_2, batch_qpred_2, batch_Q1Dpred_2, lam, dt, Batch, n)
                batch_qpred_4, batch_Q1Dpred_4, batch_upred_4  = model(batch_upred_3, batch_qpred_3, batch_Q1Dpred_3, lam, dt, Batch, n)
                batch_qpred_5, batch_Q1Dpred_5, batch_upred_5  = model(batch_upred_4, batch_qpred_4, batch_Q1Dpred_4, lam, dt, Batch, n)
                loss = L(batch_qpred_1, batch_q_1) + lambda2 * L(batch_Q1Dpred_1, batch_Q1D_1) + \
                       L(batch_qpred_2, batch_q_2) + lambda2 * L(batch_Q1Dpred_2, batch_Q1D_2) + \
                       L(batch_qpred_3, batch_q_3) + lambda2 * L(batch_Q1Dpred_3, batch_Q1D_3) + \
                       L(batch_qpred_4, batch_q_4) + lambda2 * L(batch_Q1Dpred_4, batch_Q1D_4) + \
                       L(batch_qpred_5, batch_q_5) + lambda2 * L(batch_Q1Dpred_5, batch_Q1D_5) + \
                       lambda3 * (L(batch_upred_2, batch_u_2) + L(batch_upred_3, batch_u_3) + L(batch_upred_4, batch_u_4))

            loss.backward()

            optimizer.step()

            scheduler.step()

            counter += 1
            Loss += loss.item()
            if i % (10 * batch_size) == 0:
                print('epoch: {}, batch-{}th, loss: {:.2E}, '.format(epoch, i // batch_size, loss.item()))
                if torch.sum(torch.isnan(loss)) > 0:
                    raise Exception("NAN error occurs!!!")

        loss_history[epoch] = Loss / counter # average loss in this iteration
        if epoch % args.print_every == 0:  
            end = time.time()
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print('epoch: {}, loss: {:.2E}, lr: {:.2E}, elapsed time: {:.2f}'.format(epoch, loss_history[epoch], lr, end - start))
        if loss<1e-15:
            break

    # save neural network
    torch.save(model.state_dict(), os.path.join(net_data_path, "net_params_U" + tauUinv + "N" + tauNinv + "Coarse" + str(args.coarse) + "steps" + str(Inner_step) + ".pkl"))



