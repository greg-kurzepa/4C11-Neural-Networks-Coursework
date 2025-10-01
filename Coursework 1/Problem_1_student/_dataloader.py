class WrappedDataLoader:
    def __init__(self, dl, func):
        """This class wraps a post_processing function `func` around dataloader `dl`, as in https://pytorch.org/tutorials/beginner/nn_tutorial.html"""
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))