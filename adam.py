import numpy as np
import torch
import optuna

def norm(A, p):
    A_abs_p = np.abs(A)**p
    sum_A = np.sum(A_abs_p, dtype=np.float64)
    return np.power(sum_A, 1/p)

def adam(A, U, S, Vt, max_iterations=500, lr=0.0013926296924013128, betas=(0.9745845335852898, 0.593182619966775), eps=2.1165747423163247e-06, weight_decay=1.4005902087699395e-09, differentiable=True):
    A_torch = torch.tensor(A.astype(np.float64), dtype=torch.float64, requires_grad=False)
    U_param = torch.nn.Parameter(torch.tensor(U.astype(np.float64), dtype=torch.float64, requires_grad=True))
    S_param = torch.nn.Parameter(torch.tensor(S.astype(np.float64), dtype=torch.float64, requires_grad=True))
    Vt_param = torch.nn.Parameter(torch.tensor(Vt.astype(np.float64), dtype=torch.float64, requires_grad=True))
    optimizer = torch.optim.Adam([U_param, S_param, Vt_param], lr, betas, eps, weight_decay, differentiable)
    p = 2.0
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        prev_U = U_param.clone().detach()
        prev_S = S_param.clone().detach()
        prev_Vt = Vt_param.clone().detach()
        current_recon_loss_l1 = torch.norm(A_torch - U_param @ S_param @ Vt_param, p=1).item()
        if abs(p - 2.0) < 1e-8:
            main_loss = torch.norm(A_torch - U_param @ S_param @ Vt_param, p='fro')
            U_orth_loss = torch.norm(U_param.t() @ U_param - torch.eye(U.shape[0], dtype=torch.float64), p='fro')
            V_orth_loss = torch.norm(Vt_param @ Vt_param.t() - torch.eye(Vt.shape[0], dtype=torch.float64), p='fro')
        else:
            main_loss = torch.norm(A_torch - U_param @ S_param @ Vt_param, p=p)
            U_orth_loss = torch.norm(U_param.t() @ U_param - torch.eye(U.shape[0], dtype=torch.float64), p=p)
            V_orth_loss = torch.norm(Vt_param @ Vt_param.t() - torch.eye(Vt.shape[0], dtype=torch.float64), p=p)

        loss = main_loss + U_orth_loss + V_orth_loss

        loss.backward()
        optimizer.step()

        new_recon_loss_l1 = torch.norm(A_torch - U_param @ S_param @ Vt_param, p=1).item()

        if new_recon_loss_l1 > current_recon_loss_l1:
            U_param.data.copy_(prev_U)
            S_param.data.copy_(prev_S)
            Vt_param.data.copy_(prev_Vt)
        p = max(p - 0.01, 1.0)

    U_updated = U_param.detach().numpy()
    S_updated = S_param.detach().numpy()
    Vt_updated = Vt_param.detach().numpy()

    return U_updated, S_updated, Vt_updated

def run_adam_with_params (size, interval, rank, error, loaded_data, lr, betas, eps, weight_decay, differentiable):
    stats = [0]*2
    # print("Adam with params:",
    #       f"lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay}, differentiable={differentiable}")
    key = f"size_{size}_interval_{interval[0]}_{interval[1]}"
    #print(key)
    U, S, Vt = loaded_data[key]
    A = U @ S @ Vt
    E = np.random.uniform(-1, 1, (size, size)) * error
    A += E
    for i in range(1,size - rank + 1):
            S[size-i,size-i] = 0
    Uu,Ss,Vvt = adam(A,S,U,Vt, 500, lr, betas, eps, weight_decay, differentiable)
    norm_l2_old = norm(A - U @ S @ Vt, 2)
    norm_l1_old = norm(A - U @ S @ Vt, 1)
    norm_l2_new = norm(A - Uu @ Ss @ Vvt, 2)
    norm_l1_new = norm(A - Uu @ Ss @ Vvt, 1)
    stats[0] = norm_l2_old/norm_l2_new
    stats[1] = norm_l1_old/norm_l1_new
    return stats
def tune_adam(sizes,intervals,rank,loaded_data):
    lr = [10,1e-1,1e-3,1e-12]
    beta1 = [0.999,0.9,0.8,0.7]
    beta2 = [0.85,0.7,0.5]
    eps = [1e-3,1e-8,1e-12]
    weight_decay = [1e-15]
    differentiable = [False,True]
    best_stats = [1,1]
    best_params = {}
    for size in sizes:
        for interval in intervals:
            for lrr in lr:
                for beta11 in beta1:
                    for beta22 in beta2:
                        for epss in eps:
                            for weight_decayy in weight_decay:
                                for diff in differentiable:
                                    # print(size,interval,rank)
                                    stats = run_adam_with_params(size,interval,rank,1e-15,loaded_data,lrr,(beta11,beta22),epss,weight_decayy,diff)
                                    if stats > best_stats:
                                        best_stats = stats
                                        best_params["lr"] = lrr
                                        best_params["beta1"] = beta11
                                        best_params["beta2"] = beta22
                                        best_params["eps"] = epss
                                        best_params["weight_decay"] = weight_decayy
                                        best_params["differentiable"] = diff
    print(best_params)


def objective(trial, sizes, intervals, rank, loaded_data):
    lr = trial.suggest_loguniform('lr', 1e-12, 1e-1)
    beta1 = trial.suggest_float('beta1', 0.7, 0.999)
    beta2 = trial.suggest_float('beta2', 0.5, 0.85)
    eps = trial.suggest_loguniform('eps', 1e-12, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-15, 1e-5)
    differentiable = trial.suggest_categorical('differentiable', [False, True])
    num_trials = 5
    total_score = 0
    for i in range(num_trials):
        best_stats = [1, 1]
        for size in sizes:
            for interval in intervals:
                stats = run_adam_with_params(size, interval, rank, 1e-15, loaded_data,
                                             lr, (beta1, beta2), eps, weight_decay, differentiable)
                if stats[1] > best_stats[1]:
                    best_stats = stats
        total_score += best_stats[1]
    return 1/(total_score/num_trials)

def tune_opt(sizes, intervals, rank, loaded_data):
    s = optuna.create_study(direction='minimize')
    s.optimize(lambda trial: objective(trial, sizes, intervals, rank, loaded_data), n_trials = 100)
    print('Best params: ', s.best_params)

def adam_descent(sizes,condnums,intervals,rank,error,loaded_data):
    stats = [0]*5
    print(loaded_data)
    print("Gradient descent")
    print(f"{'Параметры':<20}{'Старое l2':<27} {'Старое l1':<27} {'Новое l2':<27} {'Новое l1':<27}")
    for size in sizes:
        for condnum in condnums:
            for interval in intervals:
                if (loaded_data.files[0].find("cond")!= -1):
                    key = f"size_{size}_cond_{condnum}_interval_{interval[0]}_{interval[1]}"
                else:
                    key = f"size_{size}_interval_{interval[0]}_{interval[1]}"
                U, S, Vt = loaded_data[key]
                A = U @ S @ Vt
                E = np.random.uniform(-1, 1, (size, size))
                E *= error
                A += E
                for i in range(1,size - rank + 1):
                    S[size-i,size-i] = 0
                Uu, Ss, Vvt = adam(A, U, S, Vt)
                norm_l2_old = norm(A - U @ S @ Vt, 2)
                norm_l1_old = norm(A - U @ S @ Vt, 1)
                norm_l2_new = norm(A - Uu @ Ss @ Vvt, 2)
                norm_l1_new = norm(A - Uu @ Ss @ Vvt, 1)
                verdict = ""
                if (norm_l2_new > norm_l2_old and norm_l1_new < norm_l1_old):
                    verdict = "Y"
                    stats[0]+=1
                elif (norm_l2_new < norm_l2_old and norm_l1_new < norm_l1_old):
                    verdict = "BB"
                    stats[2]+=1
                elif (norm_l2_new > norm_l2_old and norm_l1_new > norm_l1_old):
                    verdict = "BW"
                    stats[3]+=1
                elif (norm_l2_new == norm_l2_old and norm_l1_new == norm_l1_old):
                    verdict = "NCH"
                    stats[4]+=1
                else:
                    verdict = "N"
                    stats[1]+=1
                n = sum(stats)
                print(f"{size:<3} {condnum:<4} {str(interval):<10} {norm_l2_old:<27.20e} {norm_l1_old:<27.20e} {norm_l2_new:<27.20e}{norm_l1_new:<27.20e} {verdict}")
                print(norm(Uu@Uu.T - np.eye(size),1), norm(np.eye(size) - Uu@Uu.T,1), norm(Vvt.T@Vvt - np.eye(size),1), norm(np.eye(size) - Vvt.T@Vvt,1))
    print("Y - " + f"{stats[0]/n*100:<4}" +"%","N - " + f"{stats[1]/n*100:<4}" +"%", "BB - " + f"{stats[2]/n*100:<4}" +"%", "BW - " + f"{stats[3]/n*100:<4}" +"%", "NCH - " + f"{stats[4]/n*100:<4}" +"%")


data = np.load('matrices_2.npz')
sizes = [3,5,10,20,50,100]
sizes1 = [3,5]
condnums = [1.01,1.2,2,5,10,50]
intervals = [(0,1),(1,1000)]
#tune_opt(sizes, intervals, 25, data)
adam_descent(sizes, condnums, intervals, 0, 1e-15, data)
