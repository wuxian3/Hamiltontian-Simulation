import torch


class mapper(object):
    def __init__(self, topology: torch.Tensor, initial_map: dict):
        self.topo = topology
        self.map = initial_map
        self.qubit_number = len(initial_map)
        self.route = []
        for i in range(self.qubit_number):
            tem = []
            for j in range(self.qubit_number):
                tem.append([])
            self.route.append(tem)
        self.router()
        print(self.route)

    def swap(self, i, j):
        r"""
        用于交换两个逻辑比特的物理位置
        :param i: 逻辑比特i
        :param j: 逻辑比特j
        :return: None
        """
        tem = self.map[i]
        self.map[i] = self.map[j]
        self.map[j] = tem

    def router(self):
        r"""
        用于计算任意物理比特i到任意物理比特j的路径
        :return:
        """
        for i in range(self.qubit_number):
            self.route[i][i].append(i)
            path_list = []
            path_list.append(self.route[i][i])
            while len(path_list):
                current_path = path_list.pop(0)
                current_pos = current_path[-1]
                for j in range(self.qubit_number):
                    if self.topo[current_pos][j] and len(self.route[i][j]) == 0:
                        new_path = current_path.copy()
                        new_path.append(j)
                        self.route[i][j] = new_path
                        path_list.append(new_path)



