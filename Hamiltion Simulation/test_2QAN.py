import numpy as np
import time
# Import 2QAN compiler passes
from py2qan import BenchArch
from py2qan import HeuristicMapper
from py2qan import QuRouter
# Import qiskit
import qiskit
from qiskit import transpile, QuantumCircuit
import qiskit.quantum_info as qi

import os
import pickle as pkl
import torch

qb_num = 4
r = 20
extra_r = 10
def gate_fidelity(a, b):
    dis = a - b
    return 1 - torch.trace(dis.conj().T @ dis)/(4 ** qb_num)

def qs_compiler(qasm, coupling_map, qaoa=True, layers=1, trials=1, mapper='qap', bgate='rzz', params=None):
    qs_circ = None
    qs_swap = (0, 0) # the number of swaps in the format (#swaps,#swaps merged with circuit gate)
    qs_g2 = 0 # the number of two-qubit gates without decomposition
    # Perform qubit mapping, routing, and scheduling only, without gate decomposition
    for trial in range(trials):
        # Both QAP and Qiskit mappers output inital qubit maps randomly,
        # one can run the mapper several times to achieve better compilation results
        # Initial qubit mapping
        start = time.time()
        hmapper = HeuristicMapper(qasm, coupling_map=coupling_map)
        if mapper == 'qap':
            # The default mapper based on Quadratic Assignment Problem
            init_map, cost = hmapper.run_qap(num_iter=200, lst_len=20)
        elif mapper == 'qiskit':
            # The mapper in Qiskit
            init_map = hmapper.run_qiskit(max_iterations=5)
        end = time.time()
        print("Mapper run time: ", end - start)
        # init_map = {circuit qubit index:device qubit index}
        print('The initial qubit map is \n', init_map)

        # Routing and scheduling, takes init_map as input
        router = QuRouter(qasm, init_map=init_map, coupling_map=coupling_map)
        if qaoa:
            # For QAOA, different layers have different gate parameters
            qs_circ0, swaps1 = router.run_qaoa(layers=layers, gammas=params[layers-1][:layers], betas=params[layers-1][layers:], msmt=True)
        else:
            # For quantum simulation circuits, we assume each layer has the same time steps
            # qs_circ0, swaps1 = router.run(layers=layers, msmt='True')
            qs_circ0, swaps1 = router.run(layers=layers, msmt=False)
        # qs_circ0 is the routed circuit without gate decomposition
        # swaps1 is a tuple=(#swaps,#swaps merged with circuit gate)

        # Two-qubit gate count and swap count
        qs_circ1 = transpile(qs_circ0, basis_gates=None, optimization_level=3)
        g2_count1 = 0
        if bgate in qs_circ1.count_ops():
            g2_count1 += qs_circ1.count_ops()[bgate]
        if 'unitary' in qs_circ1.count_ops():
            g2_count1 += qs_circ1.count_ops()['unitary']
        if 'swap' in qs_circ1.count_ops():
            g2_count1 += qs_circ1.count_ops()['swap']
        if trial == 0:
            qs_circ = qs_circ1
            qs_swap = swaps1
            qs_g2 = g2_count1
        elif g2_count1 < qs_g2:
            qs_circ = qs_circ1
            qs_swap = swaps1
            qs_g2 = g2_count1
        print(g2_count1, qs_swap)
    return qs_circ, qs_swap, qs_g2


def qiskit_decompose(circ, basis_gates=['id', 'rz', 'u3', 'u2', 'cx', 'reset'], bgate='cx'):
    # Perform gate decomposition and optimization into cx gate set
    # For decomposition into other gate sets, e.g., the SYC, sqrt iSWAP, iSWAP,
    # one can use Google Cirq for decomposition or the NuOp (https://github.com/prakashmurali/NuOp) decomposer
    decom_g2 = 0
    decom_circ = transpile(circ, basis_gates=basis_gates, optimization_level=3)
    if bgate in decom_circ.count_ops():
        decom_g2 += decom_circ.count_ops()[bgate]
    if 'unitary' in decom_circ.count_ops().keys():
        decom_g2 += decom_circ.count_ops()['unitary']
    return decom_circ, decom_g2



# Benchmarks
# qaoa = True
qaoa = False
# QAOA benchmarks
# OpenQASM circuits here only contain one layer/depth
with open(os.path.join('qaoa_qasms.pkl'), 'rb') as f:
    qasms = pkl.load(f)
# The parameters here include gammas for rzz and betas for rx in 4 layers
with open(os.path.join('qaoa_params.pkl'), 'rb') as f:
    params = pkl.load(f)

param = None
idx = -2  # circuit id
c_qasm = qasms[idx]
if qaoa:
    param = params[idx]
# print(c_qasm)
c_qasm = '''OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rx(1.57) q[0];
h q[1];
cx q[0],q[1];
rz(0.13) q[1];
cx q[0],q[1];
h q[1];
rx(-1.57) q[0];
rz(0.02) q[0];
h q[2];
cx q[0],q[2];
rz(0.1) q[2];
cx q[0],q[2];
h q[2];
rx(1.57) q[0];
cx q[0],q[3];
rz(0.1) q[3];
cx q[0],q[3];
rx(-1.57) q[0];
'''
# 用于比对的原电路
test_circ = qiskit.QuantumCircuit.from_qasm_str(c_qasm)

test_op = qi.Operator(test_circ)

g_count_test = 0
tem_bgate = 'rzz'
test_ops = test_circ.count_ops()
if tem_bgate in test_circ.count_ops():
    g_count_test += test_circ.count_ops()[tem_bgate]
if 'unitary' in test_circ.count_ops():
    g_count_test += test_circ.count_ops()['unitary']
if 'swap' in test_circ.count_ops():
    g_count_test += test_circ.count_ops()['swap']

print(g_count_test)

# Device information
# gate set, assume cx as the native two-qubit gate
basis_gates = ['id', 'rz', 'u3', 'u2', 'cx', 'reset', 'unitary']

# topology, assume grid architecture as an example
qn = len(test_circ.qubits)
dx = int(np.sqrt(qn))
print('The number of qubits is ', qn)
if dx*dx >= qn:
    lattice_xy = (dx, dx)
elif dx*(dx+1) >= qn:
    lattice_xy = (dx, dx+1)
elif dx*(dx+2) >= qn:
    lattice_xy = (dx, dx+2)
grid_topology = BenchArch(c_qasm, lattice_xy=lattice_xy).topology
coupling_map = [list(edge) for edge in list(grid_topology.edges)]
coupling_map += [[edge[1], edge[0]] for edge in list(grid_topology.edges)]

qs_circ, qs_swap, qs_g2 = qs_compiler(c_qasm, coupling_map, qaoa=False, layers=1, trials=5, bgate='rzz', params=param)
# print('The number of SWAPs: ', qs_swap, qs_g2)

#
qs_circ2, qs_g2 = qiskit_decompose(qs_circ, bgate='cx', basis_gates=basis_gates)
print('The number of CNOTs: ', qs_g2)
#
# ibm_circ, ibm_swap, ibm_g2 = qs_compiler(c_qasm, coupling_map, qaoa=qaoa, layers=1, trials=5, mapper='qiskit', bgate='rzz', params=param)
# print('The number of SWAPs: ', ibm_swap, ibm_g2)
#
# ibm_circ2, ibm_g2 = qiskit_decompose(ibm_circ, bgate='cx', basis_gates=basis_gates)
# print('The number of CNOTs: ', ibm_g2)
#
# # test_circ only has one layer in the given example
qiskit_circ = transpile(test_circ, basis_gates=basis_gates, coupling_map=coupling_map, optimization_level=3)
print('The number of CNOTs: ', qiskit_circ.count_ops()['cx'])


# print(qs_circ)
qan_op = qi.Operator(qs_circ)
qan_mat = torch.tensor(qan_op.data)
qan_mul = torch.tensor(qan_op.data)

test_mat = torch.tensor(test_op.data)
test_mul = torch.tensor(test_op.data)

qiskit_op = qi.Operator(qiskit_circ)
qiskit_mat = torch.tensor(qiskit_op.data)
qiskit_mul = torch.tensor(qiskit_op.data)
for i in range(r):
    qan_mat = qan_mat @ qan_mat
    test_mat = test_mat @ test_mul
    qiskit_mat = qiskit_mat @ qiskit_mul

print(gate_fidelity(qiskit_mat, test_mat))
print(gate_fidelity(qan_mat, test_mat))
print(gate_fidelity(qan_mat, qiskit_mat))