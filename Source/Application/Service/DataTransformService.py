import string

import torch


class DataTransformService():

    def __init__(self):
        self.dictionary = string.ascii_letters + " '-"

    def categories_to_tensor(self, _categories: list):
        categories = [category.split(".")[0] for category in _categories]
        return [torch.LongTensor([categories.index(category)]) for category in categories]

    def names_to_tensor(self, names: list):
        tensor_names = list()
        for name in names:
            tensor = torch.zeros(len(name), len(self.dictionary))
            for i, char in enumerate(name):
                tensor[i][self.dictionary.index(char)] = 1

            tensor_names.append(tensor)

        return tensor_names
