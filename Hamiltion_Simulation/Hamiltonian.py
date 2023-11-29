import math
import random
# import paddle_quantum as pq
import numpy as np

pauli_group = ['X', 'Z', 'Y', 'I']


def random_generate_Hamiltonian(qubit_number: int, pauli_string_number: int, max_local: int, custom: int = 0) -> object:
    r"""
    根据给定参数生成一组哈密顿量串，可以直接调用pq.Hamiltonian将该串转为pq.Hamiltonian
    :param qubit_number: 比特数量
    :param pauli_string_number: 泡利串数量
    :param max_local: 最大允许的local数
    :param custom: 默认0，表示完全随机，若为1，则可手动定义参数出现的概率（未实现，将来吧反正
    :return: 哈密顿量表示串
    """
    Hamiltonian_list = []
    for i in range(pauli_string_number):
        k_local = random.randint(1, max_local)
        pauli = []
        random_index = random.sample(range(0, qubit_number), k_local)
        random_index.sort()
        p_str = ''
        for k in range(k_local):
            p = random.randint(0, 2)
            p_str += pauli_group[p]
            p_str += str(random_index[k])
            p_str += ','
        if custom == 0:
            para = random.random() * 1
        else:
            para = np.random.choice([0.01, 0.2, 0.95], p=[0.79, 0.2, 0.01])
        H = (para, p_str.strip(','))

        Hamiltonian_list.append(H)
    return Hamiltonian_list
