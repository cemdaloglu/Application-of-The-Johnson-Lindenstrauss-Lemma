import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import math
import time

rand_numbers = np.random.default_rng(seed=0)


def x_generator(m, n):
    return rand_numbers.random((m, n))


def k_generator(X, eps):
    m, n = X.shape
    return 8 * np.log(n) / (eps ** 2)


def J_generator(X, k):
    m, _ = X.shape
    return rand_numbers.normal(0, 1 / math.sqrt(k), size=(m, k))


def condition_check(X, J, eps):
    _, n = X.shape
    check = n * n - n
    gpu_x = cp.asarray(X)
    x_gpu_norm = gpu_x.T.dot(gpu_x)
    x_norm = cp.asnumpy(x_gpu_norm)
    V = J.T.dot(X)
    Jx = V.T.dot(V)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = Jx[i, i] - 2 * Jx[i, j] + Jx[j, j]
                norm_ = x_norm[i, i] - 2 * x_norm[i, j] + x_norm[j, j]
                if not (dist <= (1 + eps) * norm_) & (dist >= (1 - eps) * norm_):
                    check = check - 1
    check = (check / (n * n - n)) * 100
    return check


m = [1000, 5000, 10000]
eps = [0.1, 0.3, 0.7, 0.9]
n_arr = [1, 1/10, 1/100]
vi = np.zeros(10)
vav = np.zeros(36)

for i in range(len(m)):
    start_time = time.time()
    for n_i in range(len(n_arr)):
        n = int(m[i] * n_arr[n_i])
        X = x_generator(m[i], n)
        for e in range(len(eps)):
            k = k_generator(X, eps[e])
            k = int(k) + 1
            print(f'k = {k}')
            for realization in range(10):
                J = J_generator(X, k)
                vi[realization] = condition_check(X, J, eps[e])
            vav[12 * i + 4 * n_i + e] = sum(vi) / 10
    plt.plot(vav[12 * i: 12 * i + 4 * 1], marker='o')
    plt.plot(vav[12 * i + 4 * 1: 12 * i + 4 * 2], marker='o')
    plt.plot(vav[12 * i + 4 * 2: 12 * i + 4 * 3], marker='o')
    plt.legend([f'{m[i] * n_arr[0]}', f'{m[i] * n_arr[1]}', f'{m[i] * n_arr[2]}'])
    plt.xlabel(f'epsilon = {eps[0], eps[1], eps[2], eps[3]}')
    plt.ylabel("V_av")
    plt.title(f'V_av plot for m = {m[i]}')
    plt.show()
    end_time = time.time()
    print(f'time: {end_time - start_time}')

for e in range(len(eps)):
    first_plot = [vav[e], vav[12 + e], vav[24 + e]]
    second_plot = [vav[4 + e], vav[16 + e], vav[28 + e]]
    third_plot = [vav[8 + e], vav[20 + e], vav[32 + e]]
    plt.plot(first_plot, marker='o')
    plt.plot(second_plot, marker='o')
    plt.plot(third_plot, marker='o')
    plt.legend([f'{m[i] * n_arr[0]}', f'{m[i] * n_arr[1]}', f'{m[i] * n_arr[2]}'])
    plt.xlabel(f'm = {m[0], m[1], m[2]}')
    plt.ylabel("V_av")
    plt.title(f'V_av plot for epsilon = {eps[e]}')
    plt.show()