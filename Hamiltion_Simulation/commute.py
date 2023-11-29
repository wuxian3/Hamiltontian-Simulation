import functools

import numpy as np
import paddle_quantum as pq
import torch
import scipy
import paddle
import Hamiltonian


_TOL_ERROR = 1e-06

def get_matrix_from_pauli_word(a: str, a_para: int):
    # hmat = torch.eye(1)
    hmat = paddle.eye(1)
    qubits = len(a)
    for i in range(qubits):
        if a[i] == 'I':
            hmat = paddle.kron(hmat, paddle.eye(2))
        if a[i] == 'X':
            hmat = paddle.kron(hmat, pq.gate.matrix.x_gate())
            # hmat.kron(pq.gate.matrix.x_gate())
        if a[i] == 'Y':
            hmat = paddle.kron(hmat, pq.gate.matrix.y_gate())
        if a[i] == 'Z':
            # hmat.kron(pq.gate.matrix.z_gate())
            hmat = paddle.kron(hmat, pq.gate.matrix.z_gate())

    ret_mat = scipy.linalg.expm(1j * a_para * hmat)
    return ret_mat


def pauli_commute_value(a: str, b: str, a_para: float, b_para: float) -> float:
    r"""
    计算两个泡利串是否对易
    :param a: 泡利串a
    :param b: 泡利串b
    :param a_para: 泡利串a的系数
    :param b_para: 泡利串b的系数
    :return: abs(exp^{i * a_para * a} @ exp^{i * b_para * b} -  exp^{i * b_para * b} @ exp^{i * a_para * a}).max()
    """
    det_ab = 0
    qubits = len(a)
    for i in range(qubits):
        if a[i] == 'I' or b[i] == 'I':
            continue
        if a[i] != b[i]:
            mata = get_matrix_from_pauli_word(a, a_para)
            matb = get_matrix_from_pauli_word(b, b_para)
            mat = mata @ matb - matb @ mata
            val = abs(mat).max()
            return float(val)
    mata = get_matrix_from_pauli_word(a, a_para)
    matb = get_matrix_from_pauli_word(b, b_para)
    val = abs(mata @ matb - matb @ mata).max()
    assert val < _TOL_ERROR, \
    f"期望的对易值为0，但实际为{val}"
    return det_ab


def pauli_commute_value_simple(a: str, b: str) -> int:
    r"""
    计算两个泡利串是否对易(简易版)，逃避了大量的矩阵的运算，仅从泡利串来判断是否可以对易
    :param a: 泡利串a
    :param b: 泡利串b
    :return: if 对易：0； else： 1
    """
    qubits = len(a)
    val = 1
    for i in range(qubits):
        if a[i] == 'I' or b[i] == 'I':
            continue
        if a[i] != b[i]:
            val *= -1
    if val == 1:
        assert pauli_commute_value(a, b, 1, 1) == 0
        return 0
    return 1


def _my_test():
    H_j = Hamiltonian.random_generate_Hamiltonian(4, 10, 4)
    H = pq.hamiltonian.Hamiltonian(H_j)
    h_coe = H.coefficients
    h_word = H.pauli_words
    print(H)
    for i in range(H.n_terms):
        for j in range(H.n_terms):
            a = h_word[i]
            b = h_word[j]
            print(a, b)
            print(pauli_commute_value(a, b, h_coe[i], h_coe[j]))


def _my_cmp(a, b):
    value_a = a[1]
    value_b = b[1]
    if value_a > value_b:
        return 1
    elif value_a < value_b:
        return -1
    return 0


def commute_number_sort(commute_list) -> list:
    r"""
    给定一个列表size=[pauli_terms, pauli_terms]，给出一个根据对易值大小进行排序的list
    :param commute_list: 给定的对应表
    :return: 一个Permutation list
    """
    ret_list = []
    n_terms = len(commute_list)
    for i in range(n_terms):
        commute_value_sum = 0
        for j in range(n_terms):
            commute_value_sum += commute_list[i][j]
        ret_list.append([i, commute_value_sum])
    ret_list.sort(key=functools.cmp_to_key(_my_cmp))
    return ret_list

# if __name__ == '__main__':
#     _my_test()