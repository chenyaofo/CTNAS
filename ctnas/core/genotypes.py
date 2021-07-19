import json

from core.dataset.architecture.common_nas import PRIMITIVES


class Genotype:
    '''
    Genotype for DARTS search space.
    '''

    def __init__(self, named_normal_arch=None, ordinal_normal_arch=None, normal_concat=None,
                 named_reduced_arch=None, ordinal_reduced_arch=None, reduced_concat=None,
                 primitives=None, loose_end=None):
        self.named_normal_arch = named_normal_arch
        self.ordinal_normal_arch = ordinal_normal_arch
        self.normal_concat = normal_concat

        self.named_reduced_arch = named_reduced_arch
        self.ordinal_reduced_arch = ordinal_reduced_arch
        self.reduced_concat = reduced_concat

        self.primitives = primitives
        self.loose_end = loose_end

    @property
    def n_nodes(self):
        return len(self.named_normal_arch) // 2

    @staticmethod
    def from_named_arch(named_normal_arch=None,  normal_concat=None,
                        named_reduced_arch=None, reduced_concat=None,
                        primitives=PRIMITIVES, loose_end=False):
        genotype = Genotype()
        assert len(named_normal_arch) == len(named_reduced_arch)
        assert len(named_normal_arch) % 2 == 0

        genotype.named_normal_arch = named_normal_arch
        genotype.normal_concat = normal_concat
        genotype.named_reduced_arch = named_reduced_arch
        genotype.reduced_concat = reduced_concat
        genotype.primitives = primitives

        genotype.ordinal_normal_arch = [(primitives.index(name), f_node, t_node)
                                        for name, f_node, t_node in named_normal_arch]
        genotype.ordinal_reduced_arch = [(primitives.index(name), f_node, t_node)
                                         for name, f_node, t_node in named_reduced_arch]
        genotype.loose_end = loose_end
        genotype._generate_concat()
        return genotype

    @staticmethod
    def from_ordinal_arch(ordinal_normal_arch=None, normal_concat=None,
                          ordinal_reduced_arch=None, reduced_concat=None,
                          primitives=PRIMITIVES, loose_end=False):
        genotype = Genotype()
        assert len(ordinal_normal_arch) == len(ordinal_reduced_arch)
        assert len(ordinal_normal_arch) % 2 == 0

        genotype.ordinal_normal_arch = ordinal_normal_arch
        genotype.normal_concat = normal_concat
        genotype.ordinal_reduced_arch = ordinal_reduced_arch
        genotype.reduced_concat = reduced_concat
        genotype.primitives = primitives

        genotype.named_normal_arch = [(primitives[index], f_node, t_node)
                                      for index, f_node, t_node in ordinal_normal_arch]
        genotype.named_reduced_arch = [(primitives[index], f_node, t_node)
                                       for index, f_node, t_node in ordinal_reduced_arch]
        genotype.loose_end = loose_end
        genotype._generate_concat()
        return genotype

    @staticmethod
    def from_arch(normal_arch=None, normal_concat=None,
                  reduced_arch=None, reduced_concat=None):
        return Genotype.from_named_arch(named_normal_arch=normal_arch,
                                        normal_concat=normal_concat,
                                        named_reduced_arch=reduced_arch,
                                        reduced_concat=reduced_concat)

    def _generate_concat(self):
        if self.loose_end:
            raise NotImplementedError()
        else:
            self.normal_concat = list(
                range(2, len(self.ordinal_normal_arch)//2+2))
            self.reduced_concat = list(
                range(2, len(self.ordinal_reduced_arch)//2+2))

    def __str__(self):
        return f"normal_arch={self.named_normal_arch},normal_concat={self.normal_concat}," + \
            f"reduced_arch={self.named_reduced_arch},reduced_concat={self.reduced_concat}"

    @staticmethod
    def from_string(arch: str):
        arch = json.loads(arch)
        genotype = Genotype(None, None, None, None)
        genotype.__dict__.update(arch)
        return genotype

    @staticmethod
    def lstm_output_to_ordinal(n_nodes, prev_nodes, prev_ops):
        assert len(prev_nodes) == 2 * n_nodes
        assert len(prev_ops) == 2 * n_nodes
        arch_list = []
        for i in range(n_nodes):
            t_node = i + 2
            f1_node = prev_nodes[i * 2]
            f2_node = prev_nodes[i * 2 + 1]
            f1_op = prev_ops[i * 2]
            f2_op = prev_ops[i * 2 + 1]
            arch_list.append((f1_op, f1_node, t_node))
            arch_list.append((f2_op, f2_node, t_node))
        return arch_list
