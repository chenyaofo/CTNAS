import json

from .base import DataBase


class MBSpaceDataBase(DataBase):
    def __init__(self):
        self.items = []

    @staticmethod
    def from_file(path):
        with open(path, "r") as f:
            raw_archs = json.load(f)
        database = MBSpaceDataBase()
        database.items = raw_archs
        return database

    @property
    def size(self):
        return len(self.items)
