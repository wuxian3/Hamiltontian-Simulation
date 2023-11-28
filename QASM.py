from typing import List, Optional, Union, Dict 

import numpy as np
import paddle_quantum as pq
from qiskit import QuantumCircuit, transpile, BasicAer
# from qiskit.providers import BaseBackend
from qiskit.providers.backend import Backend

from sample import get_a_score

class QASM_qdrift:
    def __init__(self, qubits: int, H: pq.hamiltonian.Hamiltonian) -> None:
        self.num_qubits = qubits
        self.H = H
        self.qasm_str = None

    def get_qasm_from_sample(self, sample_list: List[int], swap_accuracy: float) -> str:
        """ get the QASM of qdrift cirucit with H and sample_list and swap_accuracy.

        Args:
            sample_list (List[int]): The list of items sampled by qdrift. Note that smaple_list[i]==n means the nth item in the string, and the index is n-1.
            swap_accuracy (float): The tolerance level for exchange of string items allowed.

        Returns:
            str: QASM representation of qdrift circuit before compilation.
        """
        # The sample is the sample number not the index.
        
        qasm = f'''OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{self.num_qubits}];\ncreg c[{self.num_qubits}];\n'''

        h_j = np.abs(np.array(self.H.coefficients))  # 获取系数
        lamda = h_j.sum()
        p_j = h_j / lamda  # 计算离散概率分布

        h_pauli_words = self.H.pauli_words
        last_pauli_word = ''
        for _ in range(self.num_qubits):
            last_pauli_word += 'I'
        for i in range(len(sample_list)):
            si = sample_list[i]
            min_cost = get_a_score(last_pauli_word, h_pauli_words[si - 1])
            min_index = i

            # 在概率相差不大的采样里挑一个开销最小的
            for j in range(i + 1, len(sample_list)):
                sj = sample_list[j]
                if abs(p_j[si - 1] - p_j[sj - 1]) > swap_accuracy:
                    continue
                sj_cost = get_a_score(last_pauli_word, h_pauli_words[sj - 1])
                if sj_cost < min_cost:
                    min_cost = sj_cost
                    min_index = j

            # 交换开销最小的采样与当前的采样
            tem = sample_list[i]
            sample_list[i] = sample_list[min_index]
            sample_list[min_index] = tem
            last_pauli_word = h_pauli_words[sample_list[i] - 1]

            # 转换为QASM字符串形式
            si = sample_list[i]
            local_index = []
            gate_his = []
            pauli_word = h_pauli_words[si - 1]
            # 预处理，对X加hadamard，对Y加旋转门
            for j in range(self.num_qubits):
                if pauli_word[j] != 'I':
                    local_index.append(j)
                    if pauli_word[j] == 'X':
                        q_str = f'h q[{j}];\n'
                        qasm += q_str
                        gate_his.append(q_str)
                    elif pauli_word[j] == 'Y':
                        q_str = f'rx({np.pi/-2}) q[{j}];\n'
                        qasm += f'rx({np.pi/2}) q[{j}];\n'
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
    
    def get_complied_qasm(self,
                          qasm: Optional[str] = None,
                          backend: Optional[Backend] = None,
                          basis_gates: Optional[List[str]] = None,
                          coupling_map: Optional[List[List[int]]] = None,
                          initial_layout: Optional[Union[Dict, List]] = None,
                          layout_method: Optional[str] = None,
                          routing_method: Optional[str] = None,
                          translation_method: Optional[str] = None,
                          scheduling_method: Optional[str] = None,
                          seed_transpiler: Optional[int] = None,
                          optimization_level: Optional[int] = None,
                          ) -> str:
        """ Compile circuit with the qiskit and get the QASM. For detail information, please check qiskit.transpile
        Args:
            qasm: Optional[str] = None: QASM of the cirucit.
            backend: If set, transpiler options are automatically grabbed from
                ``backend.configuration()`` and ``backend.properties()``.
                If any other option is explicitly set (e.g., ``coupling_map``), it
                will override the backend's.

            basis_gates (Optional[List[str]], optional): List of basis gate names to unroll to
                (e.g: ``['u1', 'u2', 'u3', 'cx']``). If ``None``, do not unroll.
            coupling_map (Optional[List[List[int]]], optional): Coupling map (perhaps custom) to target in mapping.
                #. List, must be given as an adjacency matrix, where each entry
                    specifies all two-qubit interactions supported by backend,
                    e.g: ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``
            initial_layout (Optional[Union[Dict, List]], optional): Initial position of virtual qubits on physical qubits.

            layout_method: Name of layout selection pass ('trivial', 'dense', 'noise_adaptive', 'sabre')
            routing_method: Name of routing pass ('basic', 'lookahead', 'stochastic', 'sabre', 'none')
            translation_method: Name of translation pass ('unroller', 'translator', 'synthesis')
            scheduling_method: Name of scheduling pass.
                * ``'as_soon_as_possible'``: Schedule instructions greedily, as early as possible
                on a qubit resource. (alias: ``'asap'``)
                * ``'as_late_as_possible'``: Schedule instructions late, i.e. keeping qubits
                in the ground state when possible. (alias: ``'alap'``)
                If ``None``, no scheduling will be done.
            seed_transpiler: Sets random seed for the stochastic parts of the transpiler
            optimization_level: How much optimization to perform on the circuits.
                Higher levels generate more optimized circuits,
                at the expense of longer transpilation time.
                * 0: no optimization
                * 1: light optimization
                * 2: heavy optimization
                * 3: even heavier optimization
                If ``None``, level 1 will be chosen as default.

        Returns:
            str: QASM of the compiled circuit
        """

        if qasm == None:
            self.qasm_str = self.get_qasm_from_sample()
        else:
            self.qasm_str = qasm
            # raise("Please run the get_qasm_from_sample before run the get_transform_qasm.")
        
        # 通过QASM字符串创建QuantumCircuit对象
        circuit = QuantumCircuit.from_qasm_str(self.qasm_str)

        # 如果没有输入后端且相应的后端限制，则使用默认的qasm_simulator模拟器后端。
        if backend == None and coupling_map == None:
            backend = BasicAer.get_backend('qasm_simulator')
            
        # 进行电路编译（在这种情况下，不会进行太多优化）
        compiled_circuit = transpile(circuit, backend = backend, basis_gates=basis_gates, coupling_map=coupling_map, initial_layout=initial_layout,
                                     layout_method=layout_method, routing_method=routing_method, 
                                     translation_method=translation_method, scheduling_method=scheduling_method, 
                                     seed_transpiler=seed_transpiler, optimization_level=optimization_level)
        
        # 转化为QASM
        qasm_str = compiled_circuit.qasm()
        
        # print(f"circuit.depth: {compiled_circuit.depth()}, gate number: {compiled_circuit.count_ops()}")
        return qasm_str

def get_sample_list(H: pq.hamiltonian.Hamiltonian, t: int=1, extra_gate_number: int=0, accuracy: int=0.1) -> List[int]:
    """ get the sample list by qdrift.

    Args:
        H (pq.hamiltonian.Hamiltonian): 哈密顿量
        t (int, optional): 模拟时间 Defaults to 1.
        extra_gate_number (int, optional): 额外增加的采样次数 Defaults to 0.
        accuracy (int, optional): 精度. Defaults to 0.1.

    Returns:
        List[int]: 对应的采样列表, int 表示采样第n个。
    """
    h_j = abs(np.array(H.coefficients))  # 获取系数
    lamda = h_j.sum()

    p_j = h_j / lamda  # 计算离散概率分布
    gate_counts = int(np.ceil(2 * lamda**2 * t**2 / accuracy) + extra_gate_number)
    sample_list = np.random.choice(a=range(1, len(p_j) + 1), size=gate_counts, replace=True, p=p_j)

    return sample_list


if __name__ == "__main__":
    # Example of the QASM
    from Hamiltonian import random_generate_Hamiltonian
    import warnings

    # set basic data
    warnings.filterwarnings("ignore")   # 隐藏 warnings
    np.set_printoptions(suppress=True, linewidth=np.nan)        # 启用完整显示，便于在终端 print 观察矩阵时不引入换行符
    pq.set_backend('density_matrix')    # 使用密度矩阵表示
    np.random.seed(100)
    
    # get Hamiltonian
    
    ## init
    qubits = 5   # 设置量子比特数
    pauli_str_num = 50   # pauli串数量
    global_local = 5 # 允许的哈密顿量阶数
    
    ## random Hamiltonian
    H_j = random_generate_Hamiltonian(qubits, pauli_str_num, global_local, custom=1)
    H = pq.hamiltonian.Hamiltonian(H_j)
    
    # get sample_list
    t = 1 # 模拟时间
    extra_gate_number = 0 # 额外采集的门数量
    accuracy = 0.1 # 结果精度
    sample_list = get_sample_list(H, t, extra_gate_number, accuracy)
    
    # get QASM
    swap_accuracy = 0.01 / pauli_str_num # 允许swap的精度
    qasm_qdrift = QASM_qdrift(qubits, H)
    qasm = qasm_qdrift.get_qasm_from_sample(sample_list=sample_list, swap_accuracy=swap_accuracy)
    
    # 具体支持了Qiskit transpile的对应接口
    qasm = qasm_qdrift.get_complied_qasm(qasm=qasm)
    
    print("QASM of the compiled circuit :\n ", qasm)
    
