import torch.nn as nn
import functools
import torch.nn.functional as F
import torch

from core.dataset.architecture.mobilespace import *


class MBSpaceController(nn.Module):
    def __init__(self, n_conditions=1, n_unit=N_UNITS,
                 depths=DEPTHS, kernel_sizes=KERNEL_SIZES, expand_ratios=EXPAND_RATIOS,
                 hidden_size=64, batch_size=1, device="cpu"):
        super(MBSpaceController, self).__init__()
        self.n_unit = n_unit
        self.depths = depths
        self.expand_ratios = expand_ratios
        self.kernel_sizes = kernel_sizes

        self.hidden_size = hidden_size

        self.condition_embedding = nn.Embedding(n_conditions, self.hidden_size)

        self.depth_embedding = nn.Embedding(len(self.depths), self.hidden_size)
        self.ratio_embedding = nn.Embedding(len(self.expand_ratios), self.hidden_size)
        self.ks_embedding = nn.Embedding(len(self.kernel_sizes), self.hidden_size)

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.depth_linear = nn.Linear(self.hidden_size, len(self.depths))
        self.width_linear = nn.Linear(self.hidden_size, len(self.expand_ratios))
        self.ks_linear = nn.Linear(self.hidden_size, len(self.kernel_sizes))

        self.batch_size = batch_size
        self.device = device
        self.reset_parameters()

    def reset_parameters(self, init_range=0.1):
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    @functools.lru_cache(maxsize=128)
    def _zeros(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), device=self.device, requires_grad=False)

    def _impl(self, probs):
        m = torch.distributions.Categorical(probs=probs)
        action = m.sample().view(-1)
        select_log_p = m.log_prob(action)
        entropy = m.entropy()
        return action, select_log_p, entropy

    def forward(self, condition=None, force_uniform=False):
        log_ps = []
        entrpys = []
        if condition is None:
            inputs = self._zeros(self.batch_size)
        else:
            inputs = self.condition_embedding(condition)

        hidden = self._zeros(self.batch_size), self._zeros(self.batch_size)
        embed = inputs

        depths = []
        ks = []
        ratios = []

        for unit in range(self.n_unit):
            # depth
            if force_uniform:
                logits = torch.zeros(len(self.depths))
            else:
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                logits = self.depth_linear(hx)
            probs = F.softmax(logits, dim=-1)
            depth, log_p, entropy = self._impl(probs)
            log_ps.append(log_p)
            entrpys.append(entropy)

            depths.append(self.depths[depth.item()])

            embed = self.depth_embedding(depth)

            for _ in range(max(self.depths)):
                # expand ratio
                if force_uniform:
                    logits = torch.zeros(len(self.expand_ratios))
                else:
                    hx, cx = self.lstm(embed, hidden)
                    hidden = (hx, cx)
                    logits = self.width_linear(hx)
                probs = F.softmax(logits, dim=-1)
                ratio, log_p, entropy = self._impl(probs)
                log_ps.append(log_p)
                entrpys.append(entropy)

                ratios.append(self.expand_ratios[ratio.item()])

                embed = self.ratio_embedding(ratio)

                # kernel_size
                if force_uniform:
                    logits = torch.zeros(len(self.kernel_sizes))
                else:
                    hx, cx = self.lstm(embed, hidden)
                    hidden = (hx, cx)
                    logits = self.ks_linear(hx)
                probs = F.softmax(logits, dim=-1)
                k, log_p, entropy = self._impl(probs)
                log_ps.append(log_p)
                entrpys.append(entropy)

                ks.append(self.kernel_sizes[k.item()])

                embed = self.ks_embedding(k)

        return arch2str(MBArchitecture(depths, ks, ratios)), sum(log_ps), sum(entrpys)


def str2arch(string):
    def split(items, separator=","):
        return [int(item) for item in items.split(separator)]
    depths_str, ks_str, ratios_str = string.split(":")
    return MBArchitecture(split(depths_str), split(ks_str), split(ratios_str))


def arch2str(arch: MBArchitecture):
    def join(items, separator=","):
        return separator.join(map(str, items))
    return f"{join(arch.depths)}:{join(arch.ks)}:{join(arch.ratios)}"
