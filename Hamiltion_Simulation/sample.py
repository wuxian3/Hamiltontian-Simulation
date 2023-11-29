import math
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import paddle_quantum as pq


# 根据采样结果生成对应的矩阵
def generate_matrix_from_sample(sample_list, H: pq.hamiltonian.Hamiltonian, t=1):
    h_j = abs(np.array(H.coefficients))  # 获取系数
    lamda = h_j.sum()
    qubits = H.n_qubits
    gate_counts = len(sample_list)
    tau = 1j * lamda * t / gate_counts
    simulation = np.identity(2 ** qubits)  # 生成单位矩阵
    for i in sample_list:
        p_term = H.terms[i-1]
        p_str = ''
        for p in p_term:
            p_str += p + ','
        p_str = p_str.strip(',')
        pauli_str_j = (1.0, p_str)  # 获取H_j，注意，应抛弃其原有系数
        H_i = pq.hamiltonian.Hamiltonian([pauli_str_j]).construct_h_matrix(qubit_num=qubits)
        simulation = np.matmul(scipy.linalg.expm(tau * H_i), simulation)
    return simulation


# 根据采样结果生成对应的矩阵（2阶
def generate_matrix_from_sample_2nd_order(sample_list, H: pq.hamiltonian.Hamiltonian, t=1):
    h_j = abs(np.array(H.coefficients))  # 获取系数
    lamda = h_j.sum()
    qubits = H.n_qubits
    gate_counts = len(sample_list)
    tau = 1j * lamda * t / gate_counts / 2
    simulation = np.identity(2 ** qubits)  # 生成单位矩阵
    for i in sample_list:
        p_term = H.terms[i-1]
        p_str = ''
        for p in p_term:
            p_str += p + ','
        p_str = p_str.strip(',')
        pauli_str_j = (1.0, p_str)  # 获取H_j，注意，应抛弃其原有系数
        H_i = pq.hamiltonian.Hamiltonian([pauli_str_j]).construct_h_matrix(qubit_num=qubits)
        simulation = np.matmul(scipy.linalg.expm(tau * H_i), simulation)

    tem_sample_list = list(sample_list)
    tem_sample_list.reverse()
    for i in tem_sample_list:
        p_term = H.terms[i - 1]
        p_str = ''
        for p in p_term:
            p_str += p + ','
        p_str = p_str.strip(',')
        pauli_str_j = (1.0, p_str)  # 获取H_j，注意，应抛弃其原有系数
        H_i = pq.hamiltonian.Hamiltonian([pauli_str_j]).construct_h_matrix(qubit_num=qubits)
        simulation = np.matmul(scipy.linalg.expm(tau * H_i), simulation)

    return simulation


def get_a_score(pauli_word_a: str, pauli_word_b: str):
    gate_cost = 0
    qubits = len(pauli_word_a)
    if pauli_word_a != pauli_word_b:
        local_num = 0
        pauli_word_match_number = 0
        for j in range(qubits):
            if pauli_word_b[j] == 'I':
                continue
            elif pauli_word_b[j] == 'Z':
                local_num += 1
                if pauli_word_a[j] == 'Z':
                    pauli_word_match_number += 1
            else:
                local_num += 1
                gate_cost += 2
                if pauli_word_b[j] == pauli_word_a[j]:
                    pauli_word_match_number += 1
                    gate_cost -= 2
        gate_cost += 2 * (local_num - 1) + 1
    return gate_cost


# 这个函数用于正常计算门开销，只对相邻的泡利串做简单优化
def get_gate_cost(sample_list, H: pq.hamiltonian.Hamiltonian):
    h_pauli_words = H.pauli_words
    gate_cost = 0
    qubits = H.n_qubits
    last_pauli_word = ''
    for _ in range(qubits):
        last_pauli_word += 'I'
    for i in sample_list:
        sample_pauli_word = h_pauli_words[i - 1]
        # 如果和上一个完全一致，那可以直接合并成一个
        if last_pauli_word == sample_pauli_word:
            continue
        # 用于记录相邻的pauli word的相似程度，并根据相似程度适当减少门的开销
        pauli_word_match_number = 0
        local_num = 0
        for j in range(qubits):
            if sample_pauli_word[j] == 'I':
                continue
            elif sample_pauli_word[j] == 'Z':
                local_num += 1
                if last_pauli_word[j] == 'Z':
                    pauli_word_match_number += 1
            else:
                local_num += 1
                gate_cost += 2
                if last_pauli_word[j] == sample_pauli_word[j]:
                    pauli_word_match_number += 1
                    gate_cost -= 2
        gate_cost += 2 * (local_num - 1) + 1
        # 根据匹配数减少CNOT门的数量 (需要考虑匹配，暂时不能直接使用)
        # gate_cost -= 2 * (pauli_word_match_number - 1)

        last_pauli_word = sample_pauli_word
    return gate_cost


# 这个函数计算门开销时，如果两个取样的概率差值足够小，则允许运行交换两个取样的顺序
def get_gate_cost_b(sample_list, H: pq.hamiltonian.Hamiltonian, swap_accuracy):
    h_j = abs(np.array(H.coefficients))  # 获取系数
    lamda = h_j.sum()
    p_j = h_j / lamda  # 计算离散概率分布
    qubits = H.n_qubits
    h_pauli_words = H.pauli_words
    gate_cost = 0
    last_pauli_word = ''
    for _ in range(qubits):
        last_pauli_word += 'I'
    for i in range(len(sample_list)):
        si = sample_list[i]
        min_cost = get_a_score(last_pauli_word, h_pauli_words[si - 1])
        min_index = i
        for j in range(i + 1, len(sample_list)):
            sj = sample_list[j]
            if abs(p_j[si - 1] - p_j[sj - 1]) > swap_accuracy:
                continue
            sj_cost = get_a_score(last_pauli_word, h_pauli_words[sj - 1])
            if sj_cost < min_cost:
                min_cost = sj_cost
                min_index = j
        tem = sample_list[i]
        sample_list[i] = sample_list[min_index]
        sample_list[min_index] = tem
        gate_cost += min_cost
        last_pauli_word = h_pauli_words[sample_list[i] - 1]
    return gate_cost


def get_simulation_error(sample_list, H, t=1):
    qubits = H.n_qubits
    origin = scipy.linalg.expm(1j * t * H.construct_h_matrix(qubit_num=qubits))
    qdrift_simulation = generate_matrix_from_sample(sample_list, H)
    distance = 0.5 * np.linalg.norm(origin - qdrift_simulation, ord=2)
    return distance


def get_simulation_error_2nd_order(sample_list, H, t=1):
    qubits = H.n_qubits
    origin = scipy.linalg.expm(1j * t * H.construct_h_matrix(qubit_num=qubits))
    qdrift_simulation = generate_matrix_from_sample_2nd_order(sample_list, H)
    distance = 0.5 * np.linalg.norm(origin - qdrift_simulation, ord=2)
    return distance


def get_simulation_gate_cost_b(sample_list, H, swap_accuracy=0.001):
    cost = get_gate_cost_b(sample_list, H, swap_accuracy)
    return cost


def get_simulation_gate_cost_b_2nd_order(sample_list, H, swap_accuracy=0.001):
    cost = get_gate_cost(sample_list, H)
    return cost * 2