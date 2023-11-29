from typing import List, Union, Tuple, Dict
import re

import paddle
import paddle_quantum as pq
import numpy as np

import warnings
warnings.filterwarnings("ignore")

"""
.convert_pq_circuit_to_qasm: Paddle Quantum circuit to QASM.
.convert_qasm_to_pq_circuit: QASM to Paddle Quantum circuit.
"""

# Paddle_quantum支持的门
# Paddle_quantum还支持universal_two_qubits和universal_three_qubits门，但Qiskit中不直接universal_two_qubits和universal_three_qubits门，因此这里我们也暂时不考虑。
# Paddle_quantum支持的三参数门 u3 在gate_history是u，在QASM中对应也是u.
# Paddle_quantum支持的cnot门，在QASM中是cx。
_paddle_single_gate = ["h", "s", "sdg", "t", "tdg", "x", "y", "z", "p", "rx", "ry", "rz", 'u'] # .u3 in gate_history is u
_paddle_two_gate = ["cnot", "cy", "cz", "swap", "cp", "crx", "cry", "crz", "cu", "rxx", "ryy", "rzz", "ms", "universal_two_qubits"]
_paddle_three_gate = ["cswap", "ccx", "universal_three_qubits"] 

# Paddle_quantum支持的参数门
_paddle_param0_gate_list = ["h", "s", "sdg", "t", "tdg", "x", "y", "z", "cnot", "cy", "cz", "swap", "ms", "cswap", "ccx"]
_paddle_param1_gate_list = ["p", "rx", "ry", "rz", "cp", "crx", "cry", "crz", "rxx", "ryy", "rzz"]
_paddle_param3_gate_list = ["u3", "cu"]

# QASM2 支持的门
_qasm_single_gate = ["h", "s", "sdg", "t", "tdg", "x", "y", "z", "p", "rx", "ry", "rz", 'u']
_qasm_two_gate = ["cx", "cy", "cz", "swap", "cp", "crx", "cry", "crz", "cu", "rxx", "rzz"]
_qasm_three_gate = ["cswap", "ccx"]
# ryy门和ms门在QASM2中，利用其余门表示，不直接支持
_qasm_custom_gate = ['ryy', 'ms']

# QASM2 参数门
_qasm_param0_gate_list = ["h", "s", "sdg", "t", "tdg", "x", "y", "z", "cx", "cy", "cz", "swap", "cswap", "ccx"]
_qasm_param1_gate_list = ["p", "rx", "ry", "rz", "cp", "crx", "cry", "crz", "rxx", 'ryy', "rzz", 'ms']
_qasm_param3_gate_list = ["u"]
_qasm_param4_gate_list = ["cu"]


def convert_pq_circuit_to_qasm(paddle_quantum_circ: pq.ansatz.circuit.Circuit) -> str:
    """
    Example:
        OPENQASM 2.0;
        include "qelib1.inc";
        gate ryy(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(3) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
        gate ms_3(param0) q0,q1 { rxx(3) q0,q1; }
        qreg q[3];
        creg c[3];
        h q[0];
        s q[0];
        sdg q[0];
        t q[0];
        tdg q[0];
        x q[0];
        y q[0];
        z q[0];
        p(3) q[0];
        rx(3) q[0];
        ry(3) q[0];
        rz(3) q[0];
        u(3,3,3) q[0];
        cx q[0],q[1];
        cy q[0],q[1];
        cz q[0],q[1];
        swap q[0],q[1];
        cp(3) q[0],q[1];
        crx(3) q[0],q[1];
        cry(3) q[0],q[1];
        crz(3) q[0],q[1];
        cu(3,3,3,0) q[0],q[2];
        rxx(3) q[0],q[1];
        ryy(3) q[0],q[1];
        rzz(3) q[0],q[1];
        ms_3(3) q[0],q[1];
        cswap q[0],q[1],q[2];
        ccx q[0],q[1],q[2];
    """
    qasm_str = ''
    # qasm使用的版本以及导入的相应库。
    qasm_version = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' 
    # 在QASM2中，不支持类似RYY和ms的门，需要用其他门进行定义表示。
    qasm_custom_gate = ''
    qasm_custom_gate_list = []
    # qasm表示电路的量子寄存器和经典寄存器比特数，因为paddle_quantum 不支持经典寄存器，这里默认经典寄存器数量和量子寄存器数量相同。
    qasm_bit_info = f'qreg q[{paddle_quantum_circ.num_qubits}];\ncreg c[{paddle_quantum_circ.num_qubits}];\n'
    # QASM中门执行的信息.
    qasm_gate_sequence = ''
    
    for gate_info in paddle_quantum_circ.gate_history:
        gate_name = gate_info['gate']
        gate_qubits = gate_info['which_qubits']
        gate_theta = gate_info['theta']
        gate_str, custom_gate_str  = _convert_gate_to_qasm_str(gate_name=gate_name, gate_qubits=gate_qubits, gate_theta=gate_theta)
        qasm_gate_sequence += gate_str
        if custom_gate_str not in qasm_custom_gate_list:
            qasm_custom_gate_list.append(custom_gate_str)
            qasm_custom_gate += custom_gate_str

    qasm_str = qasm_version + qasm_custom_gate + qasm_bit_info + qasm_gate_sequence
    
    return qasm_str

def convert_qasm_to_pq_circuit(qasm: str) -> pq.ansatz.circuit.Circuit:
    """将QASM转换为Paddle Quantum 的电路

    Args:
        qasm (str): QASM

    Returns:
        pq.ansatz.circuit.Circuit: Circuit
    """
    qubits_number, gate_history = _qasm_to_gate_history(qasm)
    qasm_to_pq_circ = pq.ansatz.circuit.Circuit(qubits_number)

    for info in gate_history:
        gate_name = info['gate']
        which_qubits = info['which_qubits']
        theta = info['theta']
        if gate_name == 'u':
            gate_name = 'u3'
        if gate_name in _paddle_param0_gate_list:
            # print(which_qubits)
            getattr(qasm_to_pq_circ, gate_name)(which_qubits)
        elif gate_name in _paddle_param1_gate_list:
            getattr(qasm_to_pq_circ, gate_name)(which_qubits, param = theta)
        elif gate_name in _paddle_param3_gate_list:
            # u3 in paddle_quantum's Circuit, u in paddle_quantum's history, QASM and qiskit
            getattr(qasm_to_pq_circ, gate_name)(which_qubits, param = theta)
    
    return qasm_to_pq_circ

def _convert_gate_to_qasm_str(gate_name: str, gate_qubits: Union[List[int], int], gate_theta: Union[paddle.Tensor, None]) -> Tuple[str, str]:
    gate_qasm_str = ''
    custom_gate_qasm_str = ''
    # custom_gate_qasm_list = []
    gate_theta_list = []
    # 转换为python的list
    if gate_theta is not None:
        gate_theta_list = gate_theta.numpy().tolist()
    
    if gate_name in _paddle_single_gate:
        if gate_name in _paddle_param0_gate_list:
            """
            Example: h q[0];
            """
            gate_qasm_str += f"{gate_name} q[{gate_qubits}];\n"
        elif gate_name in _paddle_param1_gate_list:
            """
            Example: p(3) q[0];
            """
            gate_qasm_str += f"{gate_name}({gate_theta_list[0]}) q[{gate_qubits}];\n"
        elif gate_name == 'u':
            """
            Example: u(3,3,3) q[0];
            """
            # u3 in paddle_quantum is equal u in qiskit
            gate_qasm_str += f"u({gate_theta_list[0]},{gate_theta_list[1]},{gate_theta_list[2]}) q[{gate_qubits}];\n"
    elif gate_name in _paddle_two_gate:
        if gate_name in _paddle_param0_gate_list:
            # ms is param0 gate in paddle_quantum but it's param1 gate in qiskit, and not in the OPENQASM2.
            if gate_name == "ms": 
                """Example
                ms_gate_name = gate ms(param0) q0,q1 { rxx(-pi/2) q0,q1; }
                gate_qasm_str = ms(-math.pi/2) q[0],q[1];
                """
                ms_gate_name = f"gate ms(param0) q0,q1" + "{ " + f"rxx(-pi/2) q0,q1;" +" }\n"
                custom_gate_qasm_str += ms_gate_name
                gate_qasm_str += f"ms(-pi/2) q[{gate_qubits[0]}],q[{gate_qubits[1]}];\n"
            elif gate_name == "cnot":
                # cnot in paddle quantum but QASM2 is the cx.
                gate_qasm_str += f"cx q[{gate_qubits[0]}],q[{gate_qubits[1]}];\n"
            else:
                """
                Example: cx q[0],q[1];
                """
                gate_qasm_str += f"{gate_name} q[{gate_qubits[0]}],q[{gate_qubits[1]}];\n"
        elif gate_name in _paddle_param1_gate_list:
            # ryy is param1 gate in paddle_quantum and qiskit, but not support in the OPENQASM2.
            if gate_name == "ryy":
                """Example
                ryy_gate_name = gate ryy_3(param0) q0,q1 { rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz(3) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1; }
                gate_qasm_str = ryy_3(3) q[0],q[1];
                """ 
                # Generate a unique ID corresponding to the parameter.
                ID = abs(int(hash(str(gate_theta_list[0]))))
                ryy_gate_name = f"gate ryy_{ID}(param0) q0,q1" + "{ " + f"rx(pi/2) q0; rx(pi/2) q1; cx q0,q1; rz({gate_theta_list[0]}) q1; cx q0,q1; rx(-pi/2) q0; rx(-pi/2) q1;" +" }\n"

                custom_gate_qasm_str += ryy_gate_name
                gate_qasm_str += f"ryy_{ID}({gate_theta_list[0]}) q[{gate_qubits[0]}],q[{gate_qubits[1]}];\n"                
            else:
                """
                Example: cp(3) q[0],q[1];
                """
                gate_qasm_str += f"{gate_name}({gate_theta_list[0]}) q[{gate_qubits[0]}],q[{gate_qubits[1]}];\n"
        elif gate_name == 'cu':
            """
            本质上cu门有四个参数, 其中第四个参数为全局相位, 这里转换默认为0.
            cu(3,3,3,0) q[0],q[2];
            """
            gate_qasm_str += f"{gate_name}({gate_theta_list[0]},{gate_theta_list[1]},{gate_theta_list[2]},0) q[{gate_qubits[0]}],q[{gate_qubits[1]}];\n"
    elif gate_name in _paddle_three_gate:
        if gate_name in _paddle_param0_gate_list:
            """
            cswap q[0],q[1],q[2];
            ccx q[0],q[1],q[2];
            """
            gate_qasm_str += f"{gate_name} q[{gate_qubits[0]}],q[{gate_qubits[1]}],q[{gate_qubits[2]}];\n"
    else:
        raise("gate_name not in gate_list")

    return gate_qasm_str, custom_gate_qasm_str

def _qasm_to_gate_history(qasm_to_paddle: str) -> Tuple[int, List[Dict[str, Union[int, List, paddle.Tensor]]]]:
    """将QASM电路转换为Paddle Quantum 的 gate_history表示.

    Args:
        qasm_to_paddle (str): QASM

    Returns:
        Tuple[int, List[Dict[str, Union[int, List, paddle.Tensor]]]]: 量子比特数, gate_history
    """
    qubits_number = 0
    gate_history = []

    lines = qasm_to_paddle.strip().split('\n')

    for line in lines:
        tokens = line.split()
        # print("tokens",tokens)
        if tokens[0] == 'qreg':
            q_reg_string = tokens[1]
            matches = re.findall(r'\[(\d+)\]', q_reg_string)
            qubits_number = int(matches[0])
        # Get gate name
        gate_name = re.split(r'[(_]', tokens[0])[0]
        theta = None
        which_qubits = None
        
        if gate_name == "cx":
            gate_name = 'cnot'
        
        if gate_name in _paddle_single_gate:
            which_qubits = int(re.findall(r'\[(\d+)\]', tokens[1])[0])
        elif gate_name in _paddle_two_gate or gate_name in _paddle_three_gate:
            which_qubits = [int(qubit_str) for qubit_str in re.findall(r'\[(\d+)\]', tokens[1])]
            
        if gate_name in _paddle_param0_gate_list:
            theta = None
        elif gate_name in _paddle_param1_gate_list:
            param_list_str = re.findall(r'\((\d.+)\)', tokens[0])
            param = paddle.create_parameter(
                        shape=[1],
                        dtype='float32',
                        default_initializer=paddle.nn.initializer.Assign(np.array(float(param_list_str[0])).reshape(1))
                        )
            theta = param.detach()
        # Because u in gate history is u, same with the QASM.
        elif gate_name in _qasm_param3_gate_list or gate_name in _qasm_param4_gate_list:
            param_list_str = re.findall(r'[-+]?\d*\.\d+|\d+', tokens[0])
            # cu gate in QASM is 4 param but only 3 in paddle quantum
            if gate_name == 'cu':
                param_list_str = param_list_str[:3]
            param_list_float = [float(param_str) for param_str in param_list_str]
            param = paddle.create_parameter(
                        shape=[3],
                        dtype='float32',
                        default_initializer=paddle.nn.initializer.Assign(np.array(param_list_float).reshape(3))
                        )
            theta = param.detach()
        if gate_name in _paddle_single_gate + _paddle_two_gate +_paddle_three_gate + _qasm_param3_gate_list:
            gate_history.append({'gate': gate_name, 'which_qubits': which_qubits, 'theta': theta})
    
    return qubits_number, gate_history

def test_convert_pq_circuit_to_qasm():
    import qiskit
    from qiskit import QuantumCircuit
    import math
    import torch

    circ = pq.ansatz.circuit.Circuit(4)
    circ.h(0)
    circ.ms([0, 1])
    circ.ccx([0, 1, 2])
    circ.cnot([1, 2])
    circ.rx(1, param=3.14)
    circ.cry([0, 2], param=3.14)
    circ.ryy([1, 2], param=3.14)
    circ.ryy([0, 1], param=3.14)
    circ.ryy([0, 1], param=3)
    circ.cswap([0, 1, 2])
    circ.cu([0,2], param=[3,3,3])
    circ.u3(0, param= [3,3,3])
    # paddle quantum circuit to QASM.
    paddle_to_qasm = convert_pq_circuit_to_qasm(circ)

    # print("paddle_to_qasm: -------------------")
    # print(paddle_to_qasm)

    # Qiskit to QASM
    qiskit_circ = QuantumCircuit(4, 4)
    qiskit_circ.h(0)
    qiskit_circ.ms(-math.pi / 2, [0, 1])
    qiskit_circ.ccx(0, 1, 2)
    qiskit_circ.cx(1, 2)
    qiskit_circ.rx(3.14, 1)
    qiskit_circ.cry(3.14, 0, 2)
    qiskit_circ.ryy(3.14, 1, 2)
    qiskit_circ.ryy(3.14, 0, 1)
    qiskit_circ.ryy(3, 0, 1)
    qiskit_circ.cswap(0, 1, 2)
    qiskit_circ.cu(3,3,3,0,0,2)
    qiskit_circ.u(3,3,3,0)
    # qiskit circuit to QASM.
    qiskit_to_qasm = qiskit_circ.qasm()

    # print("qiskit_to_qasm: -------------------")
    # print(qiskit_to_qasm)

    qiskit_circuit = QuantumCircuit.from_qasm_str(qiskit_to_qasm)
    paddle_circuit = QuantumCircuit.from_qasm_str(paddle_to_qasm)

    dis = qiskit.quantum_info.Operator(qiskit_circuit) - qiskit.quantum_info.Operator(paddle_circuit)
    dis = torch.tensor(dis.data)
    dis = abs(dis)
    assert (dis.max() < 1e-6), f"convert_pq_circuit_to_qasm error, the dis is :{dis}"

def test_convert_qasm_to_pq_circuit():
    # paddle quantum circuit to QASM.
    circ = pq.ansatz.circuit.Circuit(4)
    circ.h(0)
    circ.ms([0, 1])
    circ.ccx([0, 1, 2])
    circ.cnot([1, 2])
    circ.rx(1, param=3.14)
    circ.cry([0, 2], param=3.14)
    circ.ryy([1, 2], param=3.14)
    circ.ryy([0, 1], param=3.14)
    circ.ryy([0, 1], param=3)
    circ.cswap([0, 1, 2])
    circ.cu([0,2], param=[3,3,3])
    circ.u3(0, param= [3,3,3])
    paddle_to_qasm = convert_pq_circuit_to_qasm(circ)
    
    # From QASM to paddle_circuit circ
    pq_circ_from_qasm = convert_qasm_to_pq_circuit(paddle_to_qasm)
    
    # print("circ:" ,circ.gate_history)
    # print("---------------------------------------------------")
    # print("circ_from_qasm:", circ_from_qasm.gate_history)
    
    dis = paddle.max(paddle.abs(circ.unitary_matrix() - pq_circ_from_qasm.unitary_matrix()))
    assert (dis < 1e-6), f"convert_qasm_to_pq_circuit error, the dis is :{dis}"
    
if __name__ == "__main__":
    test_convert_pq_circuit_to_qasm()
    print("-----test_convert_pq_circuit_to_qasm Pass------")
    test_convert_qasm_to_pq_circuit()
    print("-----test_convert_qasm_to_pq_circuit Pass------")
