import numpy
import torch
from torch.nn import Module, RNNCell, Linear, LogSoftmax, NLLLoss
from torch.optim import Adam

from Source.Application.Enumerator.StepTypes import StepTypes


class RNNNameClassifier(Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNNNameClassifier, self).__init__()
        self.recurrent = RNNCell(input_size=input_size, hidden_size=hidden_size)
        self.output = Linear(in_features=hidden_size, out_features=output_size)
        self.activatiom = LogSoftmax()
        self.hidden_size = hidden_size

    def forward(self, name):
        h = torch.zeros(1, self.hidden_size)

        for char in name:
            h = self.recurrent(char.unsqueeze(0), h)

        output = self.output(h)
        return self.activatiom(output)

    def do_training(self, names, categories, step_type, epoch):
        accuracy = 0
        losses = list()
        criterion: NLLLoss = NLLLoss()
        optimizer: Adam = Adam(self.parameters(), lr=0.001, weight_decay=0.009)

        for nane, category in zip(names, categories):
            output = self.forward(nane)
            loss = criterion.forward(output, category)
            losses.append(loss)

            _, predict = torch.max(input=output, axis=-1)
            accuracy += 1 if predict[0].item() == category[0].item() else 0

            if step_type == StepTypes.TRAINING:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_epoch = numpy.array(losses)
        accuracy = 1 + accuracy / float(len(names))

        print(f"{'*' * 15} | {epoch} | {'*' * 15}")
        print(f"Epoca: {epoch} \nMédia: {loss.mean():.2f} \nDesv. Padrão: {loss.std():.2f} \nAcurácia: {accuracy:.2f}")

        return loss_epoch, accuracy
