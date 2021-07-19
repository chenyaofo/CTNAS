import json

from ..architecture import NASBenchArchitecture
from .base import DataBase


class NASBenchDataBase(DataBase):
    def __init__(self):
        self.archs = {}
        self.items = []

    @staticmethod
    def from_file(path):
        with open(path, "r") as f:
            raw_archs = json.load(f)
        archs = {}
        for raw_arch in raw_archs:
            arch = NASBenchArchitecture.from_dict(raw_arch)
            archs[arch.hash_] = arch
        database = NASBenchDataBase()
        database.archs = archs
        database.items = [v for k, v in archs.items()]
        database._sort()
        return database

    def _sort(self):
        sorted_items = []
        for hash_, arch in self.archs.items():
            sorted_items.append((hash_, arch.test_accuracy))
        sorted_items = sorted(sorted_items, key=lambda item: item[1], reverse=True)
        for i, (hash_, _) in enumerate(sorted_items, start=1):
            self.archs[hash_].rank = i

    def fetch_by_hash(self, arch_hash):
        return self.archs[arch_hash]

    def fetch_by_spec(self, arch):
        if arch.matrix is None:
            return arch
        return self.fetch_by_hash(arch.hash())

    @property
    def size(self):
        return len(self.items)
