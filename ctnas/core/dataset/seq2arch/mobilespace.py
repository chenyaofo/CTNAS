from ..architecture import MBArchitecture


def str2arch(string):
    def split(items, separator=","):
        return [int(item) for item in items.split(separator)]
    depths_str, ks_str, ratios_str = string.split(":")
    return MBArchitecture(split(depths_str), split(ks_str), split(ratios_str))


def arch2str(arch: MBArchitecture):
    def join(items, separator=","):
        return separator.join(map(str, items))
    return f"{join(arch.depths)}:{join(arch.ks)}:{join(arch.ratios)}"
