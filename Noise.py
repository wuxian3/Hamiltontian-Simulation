from qiskit import QuantumCircuit, transpile
from typing import List, Optional, Dict 
import qiskit_aer.noise as noise
from qiskit_aer.noise.noise_model import NoiseModel
from qiskit_aer import AerSimulator

#Consider single qubit gate and two qubit gate
_one_qubit_gate = set(
    [
        "u1",
        "u2",
        "u3",
        "u",
        "p",
        "r",
        "rx",
        "ry",
        "rz",
        "id",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "sx",
        "sxdg",
        "t",
        "tdg",
    ]
)
_two_qubit_gate = set(
    [
        "swap",
        "cx",
        "cy",
        "cz",
        "csx",
        "cp",
        "cu",
        "cu1",
        "cu2",
        "cu3",
        "rxx",
        "ryy",
        "rzz",
        "rzx",
        "ecr",
    ]
)
_three_qubit_gate = set(["ccx", "cswap"])

# 将所有的Noise模型定义在该类下面。
class Noise_Model:
    def __init__(self) -> None:
        self.noise_model = None
    
    def depolarizing_noise_model(self, prob_1: float= 0.001, prob_2: float= 0.01, prob_3: float= 0.1, basis_gates: Optional[List[str]]=None, qubits: Optional[List[int]]=None) -> NoiseModel:
        """ Depolarizing Noise simulation
        Args:
            prob_1: probability of a 1-qubit gate to be applied
            prob_2: probability of a 2-qubit gate to be applied
            basis_gates: list of basis gates to be simulated
            coupling_map: list of coupling map
        """
        # Depolarizing quantum errors

        error_1 = noise.depolarizing_error(prob_1, 1)
        error_2 = noise.depolarizing_error(prob_2, 2) 
        error_3 = noise.depolarizing_error(prob_3, 3)
        
        #add error in basic_gates
        noise_model = noise.NoiseModel(basis_gates=basis_gates)
        
        # 如果basic_gate 为None，则使用默认的门和错误作为示例。
        if basis_gates is None:
            basis_gates = noise_model.basis_gates
            if qubits == None:
                # 如果qubits为None，则默认作用所有比特
                noise_model.add_quantum_error(error_1, ['u1', 'u2', 'u3'])
                noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
            elif isinstance(qubits, list):
                print(basis_gates, qubits)
                noise_model.add_quantum_error(error_1, ['u1', 'u2', 'u3'], qubits)
                noise_model.add_quantum_error(error_2, ['cx'], qubits)
        else:
            for basis_gate in basis_gates:
                if qubits == None:
                    # 如果qubits为None，则默认作用所有比特
                    if basis_gate in _one_qubit_gate:
                        noise_model.add_all_qubit_quantum_error(error_1, basis_gate)
                    elif basis_gate in _two_qubit_gate:
                        noise_model.add_all_qubit_quantum_error(error_2, basis_gate)
                    elif basis_gate in _three_qubit_gate:
                        noise_model.add_all_qubit_quantum_error(error_3, basis_gate)
                    else:
                        raise ValueError('Basis gate {} not in list of basis gates'.format(basis_gate))
                elif isinstance(qubits, list):
                    if basis_gate in _one_qubit_gate:
                        noise_model.add_quantum_error(error_1, basis_gate, qubits)
                    elif basis_gate in _two_qubit_gate:
                        noise_model.add_quantum_error(error_2, basis_gate, qubits)
                    elif basis_gate in _three_qubit_gate:
                        noise_model.add_quantum_error(error_3, basis_gate, qubits)
                    else:
                        raise ValueError('Basis gate {} not in list of basis gates'.format(basis_gate))
                else:
                    raise ValueError('qubits must be a list or None')
                    
        self.noise_model = noise_model
        
        return noise_model

class Noise_simulation:
    def __init__(self, circuit_qasm: str)-> None:
        self.circuit_qasm = circuit_qasm
        self.circuit = QuantumCircuit.from_qasm_str(circuit_qasm)
    
    def transpile_circuit(self, noise_model: NoiseModel, basis_gates: Optional[List[str]]=None, coupling_map: Optional[List[List[int]]] = None) -> QuantumCircuit:
        backend = AerSimulator(noise_model=noise_model,
                            coupling_map=coupling_map,
                            basis_gates=basis_gates)
        
        transpiled_circuit = transpile(self.circuit, backend)
        return transpiled_circuit
    
    def simulate(self, noise_model: NoiseModel, basis_gates: Optional[List[str]]=None, coupling_map: Optional[List[List[int]]]=None, shots: int= 1024)-> Dict[str, int]:
        """ Simulate the circuit with noise model
        Args:
            noise_model (NoiseModel): Noise model
            basis_gates (Optional[List[str]]): Basis gates
            coupling_map (Optional[List[List[int]]]): Coupling map
            shots (int): Number of shots
        
        Returns:
            Dict[str:int]: Counts of each state
        """
        backend = AerSimulator(noise_model=noise_model,
                            coupling_map=coupling_map,
                            basis_gates=basis_gates)
        
        transpiled_circuit = transpile(self.circuit, backend)
        
        result = backend.run(transpiled_circuit, shots = shots).result()
        
        counts = result.get_counts(0)
        return counts

if __name__ == '__main__':
    circ = QuantumCircuit(3,3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.measure([0, 1, 2], [0, 1, 2])
    
    circ_qasm = circ.qasm()
    
    # Get a noise model
    noise_model = Noise_Model().depolarizing_noise_model(qubits=[1,2])
    
    # 生成一个noise模拟的实例
    noise_simlation = Noise_simulation(circ_qasm)
    
    # 传入对应限制，进行编译运行。
    result = noise_simlation.simulate(noise_model, basis_gates= None, coupling_map = None, shots=1024)
    print(result)