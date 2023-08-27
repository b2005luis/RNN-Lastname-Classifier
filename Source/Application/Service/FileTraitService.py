import os

import numpy
import unicodedata


class FileTraitService():

    def __init__(self):
        self.root_data = "./../../../Assets/Data/Names"

    def read_file(self, file_path):
        file = open(file=file_path, mode="r", encoding="utf8", errors="ignore")
        names = file.read().split('\n')
        file.close()

        names = [unicodedata.normalize('NFKD', name).encode('ascii', 'ignore') for name in names]
        category = file.split("/")[-1].split(".")[0]
        categories = numpy.repeat(category, len(names))

        return names, categories

    def read_directory(self):
        names, categories = [], []
        files = os.listdir(self.root_data)

        for file in files:
            t_names, t_categories = self.read_file(os.path.join(self.root_data, file))
            names.append(t_names)
            categories.append(t_categories)

        return names, categories

    def data_balancing(self, _names: list, _categories: list, batch_size: int = 50):
        names, categories = [], []
        for c, category in enumerate(_categories):
            data_size = len(_names[c])
            indexes = numpy.random.choice(range(data_size), size=batch_size)
            names.extend([_name for _n, _name in enumerate(_names[c]) if _n in indexes])
            categories.extend([_categury for _c, _categury in enumerate(_categories[c]) if _c in indexes ])

        return names, categories
