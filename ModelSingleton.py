from random import randint

class ModelSingleton:
    def __init__(self):
        pass

    def predict(self, img) -> int:
        return randint(0, 10)