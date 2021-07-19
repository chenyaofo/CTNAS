import json

from ..architecture import CommonNASArchitecture
from .base import DataBase


class CommonNASDataBase(DataBase):
    def __init__(self):
        self.items = []

    @staticmethod
    def from_file(path):
        with open(path, "r") as f:
            raw_archs = json.load(f)
        database = CommonNASDataBase()
        database.items = [CommonNASArchitecture.from_dict(item) for item in raw_archs]
        return database

    @property
    def size(self):
        return len(self.items)
