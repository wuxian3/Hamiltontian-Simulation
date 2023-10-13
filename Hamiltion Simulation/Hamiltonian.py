import math
import random
# import paddle_quantum as pq

pauli_group=['X', 'Z', 'Y']
pi = math.pi
class Hamiltonian:
    def __init__(self, para, pauli):
        self.para = float(para)
        self.pauli = []
        self.index = []
        self.size = len(pauli)
        for i in range(len(pauli)):
            self.pauli.append(pauli[i][0])
            self.index.append(int(pauli[i][1:]))

    def __str__(self):
        ret_str = str(self.para) + ' '
        for i in range(self.size):
            ret_str += self.pauli[i] + str(self.index[i]) + ','

        return ret_str.strip(',')

    # Such a Hamiltonian is
    # for i in range(len(self.para)):
    #   H = self.para[i] * self.pauli[i]

def random_generate_Hamiltonian(qubit_number, pauli_string_number):
    Hamiltonian_list = []
    for i in range(pauli_string_number):
        k_local = random.randint(1, 4)
        pauli = []

        random_index = random.sample(range(0, qubit_number), k_local)

        for k in range(k_local):
            p_str = ''
            p = random.randint(0, 1)
            p_str += pauli_group[p]
            p_str += str(random_index[k])
            pauli.append(p_str)

        H = Hamiltonian(random.random() * math.pi, pauli)
        Hamiltonian_list.append(H)

    return Hamiltonian_list

H_list = random_generate_Hamiltonian(5, 30)
print(H_list)
fout = open('test/input.txt', 'w')
for h in H_list:
    fout.write(str(h))
    fout.write('\n')

H_list = random_generate_Hamiltonian(5, 3)
print(H_list)
fout = open('test/obv.txt', 'w')
for h in H_list:
    fout.write(str(h))
    fout.write('\n')
