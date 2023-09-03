import os

from Source.Application.Core.RNNNameClassifier import RNNNameClassifier
from Source.Application.Enumerator.StepTypes import StepTypes
from Source.Application.Service.DataTransformService import DataTransformService
from Source.Application.Service.FileTraitService import FileTraitService


class RNNTrainingService():

    def __init__(self):
        self.file_trait = FileTraitService()
        self.data_transform = DataTransformService()
        self.network = RNNNameClassifier(input_size=len(self.data_transform.dictionary),
                                         hidden_size=256,
                                         output_size=18)

    def execute(self):
        loss_epochs = list()
        accuracies = list()
        directory_path = os.path.join(os.getcwd(), "..\\..\\..\\Assets\\Data\\Names")

        names, categories = self.file_trait.read_directory(directory_path)

        features = self.data_transform.names_to_tensor(names)
        labels = self.data_transform.categories_to_tensor(categories)

        loss_epochs, accuracies = self.network.do_training(features=features,
                                                           labels=labels,
                                                           step_type=StepTypes.TRAINING,
                                                           epochs=100)


service = RNNTrainingService()
""
print(service.execute())
