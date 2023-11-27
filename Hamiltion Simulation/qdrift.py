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
import Hamiltonian
import mapping
import sample

qubits = 5   # 设置量子比特数
pauli_str_num = 50   # pauli串数量
warnings.filterwarnings("ignore")   # 隐藏 warnings
np.set_printoptions(suppress=True, linewidth=np.nan)        # 启用完整显示，便于在终端 print 观察矩阵时不引入换行符
pq.set_backend('density_matrix')    # 使用密度矩阵表示
pauli_group = ['X', 'Z', 'Y', 'I']
accuracy = 0.1
swap_accuracy = 0.01 / pauli_str_num
t = 1
print_flag = False
global_local = qubits
extra_gate_number = 10

topology = torch.zeros(size=[qubits, qubits])


def output_simulation_gate_cost(sample_list, true_random_order, new_sample_list, throw_sample_list, H):
    qdrift_cost = sample.get_gate_cost(sample_list, H)
    true_random_cost = sample.get_gate_cost(true_random_order, H)
    part_random_cost = sample.get_gate_cost(new_sample_list, H)
    throw_cost = sample.get_gate_cost(throw_sample_list, H)
    if print_flag:
        print(f'qdrift门开销为: {qdrift_cost}')
        print(f'完全随机qdrift门开销为: {true_random_cost}')
        print(f'部分随机qdrift门开销为: {part_random_cost}')
        print(f'丢弃部分qdrift门开销为: {throw_cost}')

    return [qdrift_cost, true_random_cost, part_random_cost, throw_cost]


def get_qasm_from_sample(sample_list, H):
    qasm = f'''OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{qubits}];\ncreg c[{qubits}];\n'''

    h_j = abs(np.array(H.coefficients))  # 获取系数
    lamda = h_j.sum()
    p_j = h_j / lamda  # 计算离散概率分布

    h_pauli_words = H.pauli_words
    last_pauli_word = ''
    for _ in range(qubits):
        last_pauli_word += 'I'
    for i in range(len(sample_list)):
        si = sample_list[i]
        min_cost = sample.get_a_score(last_pauli_word, h_pauli_words[si - 1])
        min_index = i

        # 在概率相差不大的采样里挑一个开销最小的
        for j in range(i + 1, len(sample_list)):
            sj = sample_list[j]
            if abs(p_j[si - 1] - p_j[sj - 1]) > swap_accuracy:
                continue
            sj_cost = sample.get_a_score(last_pauli_word, h_pauli_words[sj - 1])
            if sj_cost < min_cost:
                min_cost = sj_cost
                min_index = j

        # 交换开销最小的采样与当前的采样
        tem = sample_list[i]
        sample_list[i] = sample_list[min_index]
        sample_list[min_index] = tem
        last_pauli_word = h_pauli_words[sample_list[i] - 1]

        si = sample_list[i]
        local_index = []
        gate_his = []
        pauli_word = h_pauli_words[si - 1]
        # 预处理，对X加hadamard，对Y加旋转门
        for j in range(qubits):
            if pauli_word[j] != 'I':
                local_index.append(j)
                if pauli_word[j] == 'X':
                    q_str = f'h q[{j}];\n'
                    qasm += q_str
                    gate_his.append(q_str)
                elif pauli_word[j] == 'Y':
                    q_str = f'rx({math.pi/-2}) q[{j}];\n'
                    qasm += f'rx({math.pi/2}) q[{j}];\n'
                    gate_his.append(q_str)

        for j in range(1, len(local_index)):
            gate_str = f'cx q[{local_index[j - 1]}],q[{local_index[j]}];\n'
            qasm += gate_str
            gate_his.append(gate_str)

        qasm += f'rz({h_j[si - 1]}) q[{local_index[-1]}];\n'

        while len(gate_his):
            gate_str = gate_his.pop(-1)
            qasm += gate_str

    return qasm


def one_time_test():
    sparse_num = int(pauli_str_num / qubits ** 2) + 1
    H_j = Hamiltonian.random_generate_Hamiltonian(qubits, pauli_str_num, global_local, custom=1)

    H = pq.hamiltonian.Hamiltonian(H_j)
    # H = H_j

    h_j = abs(np.array(H.coefficients))  # 获取系数
    lamda = h_j.sum()

    p_j = h_j/lamda  # 计算离散概率分布
    gate_counts = math.ceil(2 * lamda**2 * t**2 / accuracy) + extra_gate_number

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

    ret_list = []
    _cost_list = output_simulation_gate_cost(sample_list, true_random_order, new_sample_list, throw_sample_list, H)
    ret_list.append(_cost_list)

    # _dis_tensor = output_simulation_error(sample_list, true_random_order, new_sample_list, throw_sample_list, H)
    _dis_tensor = [sample.get_simulation_error(sample_list, H),
                   sample.get_simulation_error(true_random_order, H),
                   sample.get_simulation_error(new_sample_list, H),
                   sample.get_simulation_error(throw_sample_list, H)]
    ret_list.append(_dis_tensor)

    _cost_list_b = [sample.get_simulation_gate_cost_b_2nd_order(sample_list, H, swap_accuracy),
                    sample.get_simulation_gate_cost_b_2nd_order(true_random_order, H, swap_accuracy),
                    sample.get_simulation_gate_cost_b_2nd_order(new_sample_list, H, swap_accuracy),
                    sample.get_simulation_gate_cost_b_2nd_order(throw_sample_list, H, swap_accuracy)]
    ret_list.append(_cost_list_b)

    _dis_tensor = [sample.get_simulation_error_2nd_order(sample_list, H),
                   sample.get_simulation_error_2nd_order(true_random_order, H),
                   sample.get_simulation_error_2nd_order(new_sample_list, H),
                   sample.get_simulation_error_2nd_order(throw_sample_list, H)]
    ret_list.append(_dis_tensor)

    # print(get_qasm_from_sample(sample_list, H))

    return ret_list

#
# initial_map = {}
# for i in range(qubits):
#     initial_map[i] = i
# topology = torch.tensor([[0, 1, 0, 1],
#                          [1, 0, 0, 0],
#                          [0, 0, 0, 1],
#                          [1, 0, 1, 0]])
# a_map = mapping.mapper(topology, initial_map)


repeat_times = 1
cost_tensor = torch.zeros(size=[4])
dis_tensor = torch.zeros(size=[4])

cost_tensor_b = torch.zeros(size=[4])
dis_tensor_b = torch.zeros(size=[4])
for _ in range(repeat_times):
    ret = one_time_test()
    cost_tensor += torch.tensor(ret[0])
    dis_tensor += torch.tensor(ret[1])

    cost_tensor_b += torch.tensor(ret[2])
    dis_tensor_b += torch.tensor(ret[3])

print(f"门的平均开销为{cost_tensor / repeat_times}")
print(f"平均模拟误差为{dis_tensor / repeat_times}")

print("下面是小误差概率交换的结果")
print(f"门的平均开销为{cost_tensor_b / repeat_times}")
print(f"平均模拟误差为{dis_tensor_b / repeat_times}")
