from typing import List, Union, Tuple

import paddle
import paddle_quantum as pq


# Paddle_quantum支持的门
# Paddle_quantum还支持universal_two_qubits和universal_three_qubits门，但Qiskit中不直接universal_two_qubits和universal_three_qubits门，因此这里我们也暂时不考虑。
_single_gate = ["h", "s", "sdg", "t", "tdg", "x", "y", "z", "p", "rx", "ry", "rz", "u3"]
_two_gate = ["cnot", "cy", "cz", "swap", "cp", "crx", "cry", "crz", "cu", "rxx", "ryy", "rzz", "ms"] # "universal_two_qubits"
_three_gate = ["cswap", "ccx"] #"universal_three_qubits"

# Paddle_quantum支持的无参数门
_param0_gate_list = ["h", "s", "sdg", "t", "tdg", "x", "y", "z", "cnot", "cy", "cz", "swap", "ms", "cswap", "ccx"]

# Paddle_quantum支持的单参数门
_param1_gate_list = ["p", "rx", "ry", "rz", "cp", "crx", "cry", "crz", "rxx", "ryy", "rzz"]

# Paddle_quantum支持的三参数门
_param3_gate_list = ["u3", "cu"]

# ryy门和ms门在QASM2中，利用其余门表示，不直接支持
_custom_gate = ['ryy', 'ms']

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
    # QASM结果
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
    pass
        
def _convert_gate_to_qasm_str(gate_name: str, gate_qubits: Union[List[int], int], gate_theta: Union[paddle.Tensor, None]) -> Tuple[str, str]:
    gate_qasm_str = ''
    custom_gate_qasm_str = ''
    # custom_gate_qasm_list = []
    gate_theta_list = []
    # 转换为python的list
    if gate_theta is not None:
        gate_theta_list = gate_theta.numpy().tolist()
    
    if gate_name in _single_gate:
        if gate_name in _param0_gate_list:
            """
            Example: h q[0];
            """
            gate_qasm_str += f"{gate_name} q[{gate_qubits}];\n"
        elif gate_name in _param1_gate_list:
            """
            Example: p(3) q[0];
            """
            gate_qasm_str += f"{gate_name}({gate_theta_list[0]}) q[{gate_qubits}];\n"
        elif gate_name == 'u3':
            """
            Example: u(3,3,3) q[0];
            """
            # u3 in paddle_quantum is equal u in qiskit
            gate_qasm_str += f"u({gate_theta_list[0]},{gate_theta_list[1]},{gate_theta_list[2]}) q[{gate_qubits}];\n"
    elif gate_name in _two_gate:
        if gate_name in _param0_gate_list:
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
                # cnot in QASM2 is the cx.
                gate_qasm_str += f"cx q[{gate_qubits[0]}],q[{gate_qubits[1]}];\n"
            else:
                """
                Example: cx q[0],q[1];
                """
                gate_qasm_str += f"{gate_name} q[{gate_qubits[0]}],q[{gate_qubits[1]}];\n"
        elif gate_name in _param1_gate_list:
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
    elif gate_name in _three_gate:
        if gate_name in _param0_gate_list:
            """
            cswap q[0],q[1],q[2];
            ccx q[0],q[1],q[2];
            """
            gate_qasm_str += f"{gate_name} q[{gate_qubits[0]}],q[{gate_qubits[1]}],q[{gate_qubits[2]}];\n"
    else:
        raise("gate_name not in gate_list")

    return gate_qasm_str, custom_gate_qasm_str


def test_this_file():
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

    # paddle quantum circuit to QASM.
    paddle_to_qasm = convert_pq_circuit_to_qasm(circ)

    # print("paddle_to_qasm: -------------------")
    # print(paddle_to_qasm)

    # Qiskit to QASM
    qiskit_circ = QuantumCircuit(4, 4)
    qiskit_circ.h(0)
    qiskit_circ.ms(-math.pi / 2, [0, 1])
    qiskit_circ.ccx(0, 1, 2)
    qiskit_circ.cnot(1, 2)
    qiskit_circ.rx(3.14, 1)
    qiskit_circ.cry(3.14, 0, 2)
    qiskit_circ.ryy(3.14, 1, 2)
    qiskit_circ.ryy(3.14, 0, 1)
    qiskit_circ.ryy(3, 0, 1)
    qiskit_circ.cswap(0, 1, 2)

    # qiskit circuit to QASM.
    qiskit_to_qasm = qiskit_circ.qasm()

    # print("qiskit_to_qasm: -------------------")
    # print(qiskit_to_qasm)

    qiskit_circuit = QuantumCircuit.from_qasm_str(qiskit_to_qasm)
    paddle_circuit = QuantumCircuit.from_qasm_str(paddle_to_qasm)

    dis = qiskit.quantum_info.Operator(qiskit_circuit) - qiskit.quantum_info.Operator(paddle_circuit)
    dis = torch.tensor(dis.data)
    dis = abs(dis)
    assert dis.max() < 1e-6


if __name__ == "__main__":
    test_this_file()
    print("-----Pass------")
