import numpy as np
import random
import functools

import math
import warnings
from itertools import permutations
import paddle
import paddle_quantum as pq
import scipy
import torch
from paddle_quantum import Circuit

import Hamiltonian
import commute
import cmath

# 一些重要的全局变量
qb_num = 3  # qubit的数量
pauli_str_num = 4  # pauli串数量

r = 5  # trotter的步数
t = 10.0  # 哈密顿量的演化时间


def input_Hamiltonian(file_name='test/input.txt'):
    fin = open(file_name, 'r')
    H_list = []
    for l in fin.readlines():
        tem_l = l.strip('\n').split(' ')
        para = float(tem_l[0])

        H_list.append((para, tem_l[1]))
    return H_list


def construct_two_order_permutation(cir, H: pq.Hamiltonian, tau, r, my_permutation):
    for _ in range(r):
        pq.trotter.construct_trotter_circuit(cir, H, tau / 2, steps=1, permutation=my_permutation)
        my_permutation.reverse()
        pq.trotter.construct_trotter_circuit(cir, H, tau / 2, steps=1, permutation=my_permutation)
        my_permutation.reverse()

    return


def _my_cmp(a, b):
    value_a = a[1]
    value_b = b[1]
    if value_a > value_b:
        return 1
    elif value_a < value_b:
        return -1
    return 0


def _order_by_coe(h_coe):
    n_terms = len(h_coe)
    ret_list = []
    for i in range(n_terms):
        ret_list.append([i, h_coe[i]])
    ret_list.sort(key=functools.cmp_to_key(_my_cmp))
    return ret_list


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    warnings.filterwarnings("ignore")  # 隐藏 warnings
    np.set_printoptions(suppress=True, linewidth=np.nan)  # 启用完整显示，便于在终端 print 观察矩阵时不引入换行符
    repeat_times = 1
    fid_dis = 0
    for _ in range(repeat_times):
        # H_j = input_Hamiltonian()
        H_j = Hamiltonian.random_generate_Hamiltonian(qb_num, pauli_str_num, qb_num)
        # 搭建模拟演化电路
        pq.set_backend('state_vector')

        H = pq.hamiltonian.Hamiltonian(H_j)
        number_of_Hamiltonian = H.n_terms
        H_cir = pq.ansatz.circuit.Circuit(qb_num)
        H_cir_p: Circuit = pq.ansatz.circuit.Circuit(qb_num)
        H_cir_p2: Circuit = pq.ansatz.circuit.Circuit(qb_num)

        commute_list = torch.zeros(size=[number_of_Hamiltonian, number_of_Hamiltonian])
        h_pauli_words = H.pauli_words
        h_coe = H.coefficients
        for i in range(number_of_Hamiltonian):
            for j in range(number_of_Hamiltonian):
                # val = commute.pauli_commute_value_simple(h_pauli_words[i], h_pauli_words[j])
                val = commute.pauli_commute_value(h_pauli_words[i], h_pauli_words[j], h_coe[i], h_coe[j])
                commute_list[i][j] = val

        # print(h_pauli_words)
        print(commute_list)
        coe_permutation: list = _order_by_coe(h_coe)
        commute_permutation = commute.commute_number_sort(commute_list)
        print(commute_permutation)
        _my_permutation = []
        for i in range(H.n_terms):
            # _my_permutation.append(commute_permutation[i][0])
            _my_permutation.append(coe_permutation[i][0])

        pq.trotter.construct_trotter_circuit(circuit=H_cir, hamiltonian=H, tau=t / r, steps=r, order=2)
        # print(H_cir)
        construct_two_order_permutation(H_cir_p, H, t / r, r, _my_permutation)
        # _my_permutation.reverse()
        random.shuffle(_my_permutation)
        construct_two_order_permutation(H_cir_p2, H, t / r, r, _my_permutation)
        # construct_two_order_permutation(H_cir_p2, H, t / r, r, coe_permutation)
        # print(H_cir_p)

        construct_h = H.construct_h_matrix(qb_num)
        origin = scipy.linalg.expm(-1j * t * construct_h)  # 计算目标哈密顿量的原始电路
        H_matrix = H_cir.unitary_matrix()
        H_matrix = H_matrix.numpy()

        print("以下是暴力搜索结果")
        max_fid = 0
        max_fid_idx = 0

        min_fid = 1
        min_fid_idx = 0
        perm = permutations(list(range(H.n_terms)))
        for i in perm:
            perm_cir = pq.ansatz.circuit.Circuit(qb_num)
            construct_two_order_permutation(perm_cir, H, t / r, r, list(i))
            perm_mat = perm_cir.unitary_matrix().numpy()
            fid_perm = pq.qinfo.gate_fidelity(perm_mat, origin)

            commute_sum = 0
            for j in range(pauli_str_num):
                tem_sum = 0
                idx_val = i[j]
                for k in range(0, j):
                    idx_cmp = i[k]
                    tem_sum += commute_list[idx_cmp][idx_val]
                for k in range(j + 1, pauli_str_num):
                    idx_cmp = i[k]
                    tem_sum -= commute_list[idx_cmp][idx_val]
                commute_sum += abs(tem_sum)

            print(f'排序结果为:{i}, 保真度为{fid_perm}, 我的对易指标为{commute_sum}')

            if fid_perm > max_fid:
                max_fid = fid_perm
                max_fid_idx = i
            if fid_perm < min_fid:
                min_fid = fid_perm
                min_fid_idx = i
        print(f'最大保真度为:{max_fid}，对应的排序结果为:{max_fid_idx}')
        print(f'最小保真度为:{min_fid}，对应的排序结果为:{min_fid_idx}')
        print(f'模拟的哈密顿量为:\n{H}')

        H_p_matrix = H_cir_p.unitary_matrix()
        H_p_matrix = H_p_matrix.numpy()

        H_p_matrix2 = H_cir_p2.unitary_matrix()
        H_p_matrix2 = H_p_matrix2.numpy()

        fid = pq.qinfo.gate_fidelity(H_matrix, origin)
        fid_p = pq.qinfo.gate_fidelity(H_p_matrix, origin)
        fid_p2 = pq.qinfo.gate_fidelity(H_p_matrix2, origin)

        print(f'门保真度为: {fid:.9f}')
        print(f'排序门保真度为: {fid_p:.9f}')
        print(f'随机排序门保真度为: {fid_p2:.9f}')

        print(f'保真度提升为:{fid_p - fid:.9f}')
        print(f'两种排序的保真度差距为:{fid_p - fid_p2:.9f}')

        fid_dis = fid_p - fid
    print(f'平均保真度提升为:{fid_dis / repeat_times:.9f}')
