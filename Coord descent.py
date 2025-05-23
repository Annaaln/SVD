import numpy as np

def norm(A, p):
    A_abs_p = np.abs(A)**p
    sum_A = np.sum(A_abs_p, dtype=np.float64)
    return np.power(sum_A, 1/p)

def pack_params(U, Sigma, V):
    return np.hstack((U.flatten(), Sigma.diagonal(), V.flatten()))
def unpack_params(params, n):
    Uf = params[:n*n].reshape(n, n)
    Sigmaf = np.diag(params[n*n:n*n+n])
    Vf = params[n*n+n:].reshape(n, n)
    return Uf, Sigmaf, Vf

def loss(params,A,size,p):
    Uf, Sigmaf, Vf = unpack_params(params, size)
    A_reconstructed = Uf @ Sigmaf @ Vf.T
    return norm(A - A_reconstructed, p) + norm(Uf@Uf.T - np.eye(size), p) + norm(Vf.T@Vf - np.eye(size), p)

def adaptive_coordinate_descent(loss_func, x0, A, size, initial_step_UV=1e-14, initial_step_S=1e-6, max_iter=200):
    x = x0.copy()
    U_size = size * size
    Sigma_size = size
    V_size = size * size
    step_sizes_UV = np.full(len(x), initial_step_UV)
    step_sizes_S = np.full(len(x), initial_step_S)
    p = 1
    for i in range(max_iter):
        for j in range(len(x)):
            x_test = x.copy()
            if j < U_size:
                step_size = step_sizes_UV[j]
            elif j < U_size + Sigma_size:
                step_size = step_sizes_S[j - U_size] * x[j]
            else:
                step_size = step_sizes_UV[j - U_size - Sigma_size]
            x_test[j] += step_size
            if loss_func(x_test, A, size, p) < loss_func(x, A, size, p):
                x[j] += step_size
                if j < U_size or j >= (U_size + Sigma_size + V_size):
                    step_sizes_UV[j] *= 2
                else:
                    step_sizes_S[j - U_size] *= 2
            else:
                x_test[j] -= 2 * step_size
                if loss_func(x_test, A, size, p) < loss_func(x, A, size, p):
                    x[j] -= step_size
                    if j < U_size or j >= (U_size + Sigma_size + V_size):
                        step_sizes_UV[j] *= 2
                    else:
                        step_sizes_S[j - U_size] *= 2
                else:
                    if j < U_size or j >= (U_size + Sigma_size + V_size):
                        step_sizes_UV[j] /= 2
                    else:
                        step_sizes_S[j - U_size] /= 2

    return x

def adaptive_coordinate_descent_S(loss_func, x0, A, size, initial_step_S=1e-10, max_iter=200):
    x = x0.copy()
    step_sizes_S = np.full(len(x), initial_step_S)
    p = 1
    for i in range(max_iter):
        for j in range(len(x)):
            if size * size <= j < size * size + size:
                x_test = x.copy()
                step_size = step_sizes_S[j - size * size] * x[j]
                x_test[j] += step_size
                if loss_func(x_test, A, size, p) < loss_func(x, A, size, p):
                    x[j] += step_size
                    step_sizes_S[j - size * size] *= 2
                else:
                    x_test[j] -= 2 * step_size
                    if loss_func(x_test, A, size, p) < loss_func(x, A, size, p):
                        x[j] -= step_size
                        step_sizes_S[j - size * size] *= 2
                    else:
                        step_sizes_S[j - size * size] /= 2
    return x


def coordinate_descent(sizes,condnums,intervals, error,func, loaded_data):
    stats = [0]*5
    print("Coordinate descent")
    print(f"{'Параметры':<20}{'Старое l2':<27} {'Старое l1':<27} {'Новое l2':<27} {'Новое l1':<27}")
    for size in sizes:
        for condnum in condnums:
            for interval in intervals:
                key = f"size_{size}_cond_{condnum}_interval_{interval[0]}_{interval[1]}"
                U, S, Vt = loaded_data[key]
                A = U @ S @ Vt
                E = np.random.uniform(-1, 1, (size, size))
                E *= error
                A += E
                rank = round(size * 0.21)
                for i in range(1, size - rank + 1):
                    S[size-i, size-i] = 0
                x0 = pack_params(U, S, Vt.T)
                x = func(loss,x0,A,size)
                Uu, Ss, Vvt = unpack_params(x,size)
                Vvt = Vvt.T
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
                print(f"{size:<3} {condnum:<4} {str(interval):<10} {norm_l2_old:<27.20e} {norm_l1_old:<27.20e} {norm_l2_new:<27.20e}{norm_l1_new:<27.20e} {verdict} {norm_l1_old/norm_l1_new}")
    print("Y - " + f"{stats[0]/n*100:<4}" +"%","N - " + f"{stats[1]/n*100:<4}" +"%", "BB - " + f"{stats[2]/n*100:<4}" +"%", "BW - " + f"{stats[3]/n*100:<4}" +"%", "NCH - " + f"{stats[4]/n*100:<4}" +"%")


data = np.load('matrices.npz')
sizes = [3,5,10,20,50,100]
sizes1 = [3,5,10, 20]
condnums = [1.01,1.2,2,5,10,50]
intervals = [(0,1),(1,1000)]
coordinate_descent(sizes,condnums,intervals, 1e-14, adaptive_coordinate_descent_S,  data)

