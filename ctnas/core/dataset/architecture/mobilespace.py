from collections import namedtuple


# following the setting in OFA
N_UNITS = 5
DEPTHS = [2, 3, 4]
N_DEPTHS = len(DEPTHS)
EXPAND_RATIOS = [3, 4, 6]
N_EXPAND_RATIOS = len(EXPAND_RATIOS)
KERNEL_SIZES = [3, 5, 7]
N_KERNEL_SIZES = len(KERNEL_SIZES)

MBArchitecture = namedtuple("MBArchitecture", ["depths", "ks", "ratios"])
