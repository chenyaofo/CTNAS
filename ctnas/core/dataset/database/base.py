class DataBase:
    def __init__(self):
        self.items = list()

    @staticmethod
    def from_file(path):
        raise NotImplementedError()

    def __len__(self):
        return len(self.items)
