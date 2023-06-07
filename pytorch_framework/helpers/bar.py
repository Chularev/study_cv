from tqdm import tqdm


class Bar:
    def __init__(self, loop):
        self.map = {}
        self.loop = tqdm(loop, leave=True)

    def set(self, key, value):
        self.map[key] = value

    def __iter__(self):
        return self.loop.__iter__()

    def __next__(self):
        return self.loop.__next__()

    def __len__(self):
        return self.loop.__len__()

    def update(self):
        self.loop.set_postfix(self.map)
