import numpy as np
import random
import functools

import math
import warnings

import paddle
import paddle_quantum as pq
import scipy
import cmath

# 一些重要的全局变量
qb_num = 5  # qubit的数量

r = 30  # trotter的步数
t = 1.0 # 哈密顿量的演化时间


def qubit_map_init(n):
    qubit_map = {}
    for i in range(n):
        qubit_map[i] = i
    return qubit_map


def input_Hamiltonian(file_name='test/input.txt'):
    fin = open(file_name, 'r')
    H_list = []
    for l in fin.readlines():
        tem_l = l.strip('\n').split(' ')
        para = float(tem_l[0])

        H_list.append((para, tem_l[1]))
    return H_list


def input_obv(file_name='test/obv.txt'):
    fin = open(file_name, 'r')
    temH_list = []
    for l in fin.readlines():
        tem_l = l.strip('\n').split(' ')
        para = float(tem_l[0])

        temH_list.append((para, tem_l[1]))

    temH = pq.hamiltonian.Hamiltonian(temH_list)
    ret_obv = scipy.linalg.expm(1j * t * temH.construct_h_matrix(qubit_num=qb_num))  # 计算目标哈密顿量的矩阵表示
    return ret_obv


# 生成一个随机的密度算子rho并返回
def random_rho():
    rstate = pq.state.random_state(qb_num)
    tem_rho = rstate.ket.matmul(rstate.bra)
    return tem_rho.numpy()


def trace_Or(sum_h, obv, input_rho):
    # rstate = pq.state.random_state(qb_num)
    tr_list= []
    # tem_state = rstate.clone()
    for i in range(sum_h.n_terms):
        para = sum_h.coefficients[i]
        term = sum_h.terms[i]
        pauli_str = ''
        for p in term:
            pauli_str += p + ','
        pauli_str = pauli_str.strip(',')

        # 子泡利串形成的哈密顿量
        subH = pq.hamiltonian.Hamiltonian([(para, pauli_str)])
        hmat = subH.construct_h_matrix(qubit_num=qb_num)

        tem_rho = np.matmul(scipy.linalg.expm(1j * t * hmat), input_rho)

        sub_tr = np.trace(tem_rho)
        tr_list.append([abs(sub_tr), subH])
    return tr_list


def tr_hamiltonian_cmp(x, y):
    if x[0] > y[0]:
        return 1
    elif x[0] < y[0]:
        return -1
    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # 隐藏 warnings
    np.set_printoptions(suppress=True, linewidth = np.nan)  # 启用完整显示，便于在终端 print 观察矩阵时不引入换行符

    # 读取一个测量算子
    obv = input_obv()
    rho = random_rho()
    H_j = input_Hamiltonian()
    # 搭建模拟演化电路
    pq.set_backend('state_vector')

    H = pq.hamiltonian.Hamiltonian(H_j)
    number_of_Hamiltonian = H.n_terms
    H_cir = pq.ansatz.circuit.Circuit(qb_num)

    H_permutation = []
    for i in range(number_of_Hamiltonian):
        H_permutation.append(i)

    random.shuffle(H_permutation)
    H_permutation = np.array(H_permutation)

    pq.trotter.construct_trotter_circuit(circuit=H_cir, hamiltonian=H, tau=t/r, steps=r)

    origin = scipy.linalg.expm(1j * t * H.construct_h_matrix(qubit_num=qb_num))  # 计算目标哈密顿量的原始电路
    H_matrix = H_cir.unitary_matrix()
    H_matrix = H_matrix.numpy()
    # print(f'模拟电路矩阵为: \n {H_matrix} \n原始电路矩阵为: \n {origin}')

    # fid = pq.qinfo.gate_fidelity(pq.linalg.dagger(H_matrix), origin)
    # print(f'门保真度为: {fid:.9f}')

    tr_list = trace_Or(H, obv, rho)
    tr_list.sort(key=functools.cmp_to_key(tr_hamiltonian_cmp), reverse=True)
    # print(tr_list)
    r2 = r

    new_H_list = []
    for j in range(len(tr_list)):
        tr = tr_list[j][1]
        para = tr.coefficients[0]
        term = tr.terms[0]
        pauli_str = ''
        for p in term:
            pauli_str += p + ','
        pauli_str = pauli_str.strip(',')
        new_H_list.append((para, pauli_str))

    for less in range(H.n_terms):
        newH = pq.hamiltonian.Hamiltonian(new_H_list)

        H2_cir = pq.ansatz.circuit.Circuit(qb_num)
        pq.trotter.construct_trotter_circuit(circuit=H2_cir, hamiltonian=newH, tau=t / r2, steps=r2)

        H2_matrix = H2_cir.unitary_matrix()
        H2_matrix = H2_matrix.numpy()
        # print(f'模拟电路矩阵为: \n {H_matrix} \n原始电路矩阵为: \n {origin}')

        # fid = pq.qinfo.gate_fidelity(pq.linalg.dagger(H2_matrix), origin)
        # print(f'门保真度为: {fid:.9f}')

        rho1 = np.matmul(H_matrix, rho)
        rho1 = np.matmul(rho1, pq.linalg.dagger(H_matrix))

        rho2 = np.matmul(H2_matrix, rho)
        rho2 = np.matmul(rho2, pq.linalg.dagger(H2_matrix))

        rho_dis = rho1 - rho2

        print(less, abs(np.trace(np.matmul(obv, rho_dis))))

        del new_H_list[-1]
