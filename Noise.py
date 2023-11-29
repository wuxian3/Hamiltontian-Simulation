from qiskit import QuantumCircuit, transpile
from typing import List, Optional, Dict , Union
import qiskit_aer.noise as noise
from qiskit_aer.noise.noise_model import NoiseModel
from qiskit_aer import AerSimulator
from itertools import combinations

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
    """ Noise Model by Qiskit Aer
        QuantumError:
            :kraus_error(noise_ops[, canonical_kraus])
                Return a Kraus quantum error channel.
            :mixed_unitary_error(noise_ops)
                Return a mixed unitary quantum error channel.
            :coherent_unitary_error(unitary)
                Return a coherent unitary quantum error channel.
            :pauli_error(noise_ops) 
                Return a mixed Pauli quantum error channel.
            :depolarizing_error(param, num_qubits) 
                Return a depolarizing quantum error channel.
            :reset_error(prob0[, prob1])
                Return a single qubit reset quantum error channel.
            :thermal_relaxation_error(t1, t2, time[, ...])
                Return a single-qubit thermal relaxation quantum error channel.
            :phase_amplitude_damping_error(param_amp, ...)
                Return a single-qubit combined phase and amplitude damping quantum error channel.
            :amplitude_damping_error
                Return a single-qubit generalized amplitude damping quantum error channel.
            :phase_damping_error
                Return a single-qubit generalized phase damping quantum error channel.
        ReadoutError(probabilities, atol=1e-08):
            Example: 1-qubit:
                probabilities[0] = [P("0"|"0"), P("1"|"0")]
                probabilities[1] = [P("0"|"1"), P("1"|"1")]
            Example: 2-qubit:
                probabilities[0] = [P("00"|"00"), P("01"|"00"), P("10"|"00"), P("11"|"00")]
                probabilities[1] = [P("00"|"01"), P("01"|"01"), P("10"|"01"), P("11"|"01")]
                probabilities[2] = [P("00"|"10"), P("01"|"10"), P("10"|"10"), P("11"|"10")]
                probabilities[3] = [P("00"|"11"), P("01"|"11"), P("10"|"11"), P("11"|"11")]
    """
    def __init__(self) -> None:
        self.noise_model = None
    
    def add_one_qubit_gate_error(self, error_gate: str, error_prob: float, basis_gates: Optional[List[str]]=None, qubits: Optional[Union[List[int], int]]=None) -> NoiseModel:
        """ Add single gate error
        Args:
            error_gate: the gate to add error
            error_prob: the probability of error
            basis_gates: the basis gates
            qubits: the qubits
        """
        _one_qubit_quantum_error = ['depolarizing_error', 'reset_error', 'thermal_relaxation_error', 'phase_amplitude_damping_error',
                               'amplitude_damping_error', 'phase_damping_error']
        
        pass
    
    def add_two_qubit_gate_error(self, error_gate: str, error_prob: float, basis_gates: Optional[List[str]]=None, qubits: Optional[List[int]]=None) -> NoiseModel:
        """ Add two qubit error
        Args:
            error_gate: the gate to add error
            error_prob: the probability of error
            basis_gates: the basis gates
            qubits: the qubits
        """
        pass
    
    
    def depolarizing_noise_model(self, prob_1: float= 0.001, prob_2: float= 0.01, prob_3: float= 0.1, basis_gates: Optional[List[str]]=None, qubits: Optional[List[int]]=None) -> NoiseModel:
        """ Depolarizing Noise simulation
        Args:
            prob_1: probability of a 1-qubit gate to be applied
            prob_2: probability of a 2-qubit gate to be applied
            prob_3: probability of a 2-qubit gate to be applied
            basis_gates: list of basis gates to be simulated
            coupling_map: list of coupling map
            qubits: list of qubits
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
                noise_model.add_all_qubit_quantum_error(error_1, ['rz', 'sx','u1', 'u2', 'u3'])
                noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
            elif isinstance(qubits, list):
                print(basis_gates, qubits)
                for qubit in combinations(qubits, 1):
                    noise_model.add_quantum_error(error_1, ['rz', 'sx','u1', 'u2', 'u3'], qubit)
                for two_qubit in combinations(qubits, 2):
                    noise_model.add_quantum_error(error_2, ['cx'], two_qubit)
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
                        for qubit in qubits:
                            noise_model.add_quantum_error(error_1, basis_gate, qubit)
                    elif basis_gate in _two_qubit_gate:
                        for two_qubit in combinations(qubits, 2):
                            noise_model.add_quantum_error(error_2, basis_gate, two_qubit)
                    elif basis_gate in _three_qubit_gate:
                        for three_qubit in combinations(qubits, 3):
                            noise_model.add_quantum_error(error_3, basis_gate, three_qubit)
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
    # noise_model = Noise_Model().depolarizing_noise_model(qubits=None)
    
    print("支持的基础门：", noise_model.basis_gates)
    print("存在Noise的命令: ", noise_model.noise_instructions)
    print("存在Noise的比特: ",noise_model.noise_qubits)

    # 生成一个noise模拟的实例
    noise_simlation = Noise_simulation(circ_qasm)
    
    # 传入对应限制，进行编译运行。
    result = noise_simlation.simulate(noise_model, basis_gates= None, coupling_map = None, shots=1024)
    print(result)