import warnings

import math
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import paddle_quantum as pq
import paddle
import torch
from paddle_quantum.trotter import construct_trotter_circuit, get_1d_heisenberg_hamiltonian

qubits = 6  # 设置量子比特数
pauli_str_num = 100     # pauli串数量
warnings.filterwarnings("ignore")   # 隐藏 warnings
np.set_printoptions(suppress=True, linewidth=np.nan)        # 启用完整显示，便于在终端 print 观察矩阵时不引入换行符
pq.set_backend('density_matrix')    # 使用密度矩阵表示
pauli_group = ['X', 'Z', 'Y', 'I']
accuracy = 0.1
t = 1
print_flag = False


def random_generate_Hamiltonian(qubit_number: int, pauli_string_number: int) -> object:
    Hamiltonian_list = []
    for i in range(pauli_string_number):
        k_local = random.randint(1, qubits)
        pauli = []
        random_index = random.sample(range(0, qubit_number), k_local)
        random_index.sort()
        p_str = ''
        for k in range(k_local):
            p = random.randint(0, 2)
            p_str += pauli_group[p]
            p_str += str(random_index[k])
            p_str += ','
        para = random.random()
        para = np.random.choice([0.01, 0.2, 0.95], p=[0.79, 0.2, 0.01])
        H = (para, p_str.strip(','))

        Hamiltonian_list.append(H)
    return Hamiltonian_list


def generate_matrix_from_sample(sample_list, H):
    h_j = abs(np.array(H.coefficients))  # 获取系数
    lamda = h_j.sum()
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


def output_simulation_error(sample_list, true_random_order, new_sample_list, throw_sample_list, H) -> None:
    origin = scipy.linalg.expm(1j * t * H.construct_h_matrix(qubit_num=qubits))  # 计算目标哈密顿量的原始电路
    qdrift_simulation = generate_matrix_from_sample(sample_list, H)
    true_random_simulation = generate_matrix_from_sample(true_random_order, H)
    part_random_simulation = generate_matrix_from_sample(new_sample_list, H)
    throw_simulation = generate_matrix_from_sample(throw_sample_list, H)

    distance = 0.5 * np.linalg.norm(origin - qdrift_simulation, ord=2)

    random_distance = 0.5 * np.linalg.norm(origin - true_random_simulation, ord=2)
    new_distance = 0.5 * np.linalg.norm(origin - part_random_simulation, ord=2)

    throw_distance = 0.5 * np.linalg.norm(origin - throw_simulation, ord=2)

    # simulate_qdrift_distance = 0.5 * np.linalg.norm(qdrift_simulation - throw_simulation, ord=2)
    if print_flag:
        print(f'模拟误差为: {distance:.4f}')
        print(f'完全随机打乱顺序的模拟误差为: {random_distance:.4f}')

        print(f'部分随机打乱顺序的模拟误差为: {new_distance:.4f}')

        print(f'丢弃部分小概率采样的模拟误差为: {throw_distance:.4f}')
        print(f'丢弃部分小概率采样后与qdrift的模拟误差为: {simulate_qdrift_distance:.4f}')
    return [distance, random_distance, new_distance, throw_distance]


def output_simulation_gate_cost(sample_list, true_random_order, new_sample_list, throw_sample_list, H):
    qdrift_cost = get_gate_cost(sample_list, H)
    true_random_cost = get_gate_cost(true_random_order, H)
    part_random_cost = get_gate_cost(new_sample_list, H)
    throw_cost = get_gate_cost(throw_sample_list, H)
    if print_flag:
        print(f'qdrift门开销为: {qdrift_cost}')
        print(f'完全随机qdrift门开销为: {true_random_cost}')
        print(f'部分随机qdrift门开销为: {part_random_cost}')
        print(f'丢弃部分qdrift门开销为: {throw_cost}')

    return [qdrift_cost, true_random_cost, part_random_cost, throw_cost]


def get_gate_cost(sample_list, H):
    h_pauli_words = H.pauli_words
    gate_cost = 0
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


def one_time_test():
    sparse_num = int(pauli_str_num / qubits ** 2) + 1
    H_j = random_generate_Hamiltonian(qubits, pauli_str_num)
    # H_j = pq.qinfo.random_hamiltonian_generator(qubits, pauli_str_num)
    # H_j = pq.qinfo.random_pauli_str_generator(qubits, pauli_str_num)
    # print(H_j)

    # 将一部分的h_j缩小
    # index = list(range(pauli_str_num))
    # random.shuffle(index)
    # for i in range(sparse_num, pauli_str_num):
    #     tup = H_j[index[i]]
    #     H_j[index[i]] = (tup[0] / (qubits ** 2 + pauli_str_num), tup[1])

    # print(H_j)
    H = pq.hamiltonian.Hamiltonian(H_j)
    # H = H_j

    h_j = abs(np.array(H.coefficients))  # 获取系数
    lamda = h_j.sum()

    p_j = h_j/lamda  # 计算离散概率分布
    gate_counts = math.ceil(2 * lamda**2 * t**2 / accuracy)

    accept_error = 1 / (len(p_j))

    r = int(gate_counts/len(p_j)) + 1
    print(f'trotter 层数为{r}')
    print(f'达到 {accuracy} 的精度需要 {gate_counts} 个酉门')
    print(f'丢弃的阈值为{accept_error}')
    print(p_j)
    total_throw_pj = 0
    for p in p_j:
        if p < accept_error:
            total_throw_pj += p
    print(total_throw_pj)

    trotter_cir = pq.ansatz.circuit.Circuit()
    pq.trotter.construct_trotter_circuit(trotter_cir, H, tau=t/r, steps=r)
    # print(trotter_cir)
    trotter_unitary = trotter_cir.unitary_matrix().numpy()
    trotter_unitary = trotter_unitary.conj().T
    origin = scipy.linalg.expm(1j * t * H.construct_h_matrix(qubit_num=qubits))  # 计算目标哈密顿量的原始电路
    trotter_distance = 0.5 * np.linalg.norm(origin-trotter_unitary, ord=2)
    print(f'trotter的模拟误差为: {trotter_distance:.4f}')

    np.random.seed(666)  # 固定随机数初始位置，便于演示说明
    # gate_counts = 200
    sample_list = np.random.choice(a=range(1, len(p_j) + 1), size=gate_counts, replace=True, p=p_j)

    # print(f'qDRIFT 的模拟电路矩阵为: \n {simulation} \n原始电路矩阵为: \n {origin}')

    true_random_order = sample_list.copy()
    random.shuffle(true_random_order)

    new_sample_list = sample_list.copy()
    random_index = []
    for i in range(len(sample_list)):
        if p_j[sample_list[i] - 1] < accept_error:
            random_index.append(i)
    ori = sample_list[random_index.copy()]

    random.shuffle(random_index)
    for i in range(len(random_index)):
        new_index = random_index[i]
        new_sample_list[new_index] = ori[i]

    throw_sample_list = []
    for i in sample_list:
        if p_j[i - 1] >= accept_error:
            throw_sample_list.append(i)

    # print(len(sample_list), len(throw_sample_list))
    if print_flag:
        print(f'采样结果为:\n {sample_list}')
        print(f'完全随机打乱采样结果为:\n {true_random_order}')
        print(f'部分随机打乱采样结果为:\n {new_sample_list}')
        print(f'丢弃部分小概率采样结果为:\n {throw_sample_list}')

    ret_list = []
    _cost_list = output_simulation_gate_cost(sample_list, true_random_order, new_sample_list, throw_sample_list, H)
    ret_list.append(_cost_list)

    _dis_tensor = output_simulation_error(sample_list, true_random_order, new_sample_list, throw_sample_list, H)
    ret_list.append(_dis_tensor)

    return ret_list

repeat_times = 100
cost_tensor = torch.zeros(size=[4])
dis_tensor = torch.zeros(size=[4])
for _ in range(repeat_times):
    ret = one_time_test()
    cost_tensor += torch.tensor(ret[0])
    dis_tensor += torch.tensor(ret[1])
print(cost_tensor / repeat_times)
print(dis_tensor / repeat_times)
