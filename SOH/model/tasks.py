import numpy as np
import torch
from collections import defaultdict

from utils import to_var


class Label:
    def __init__(self, column_name, training_set, sample_length=50):
        self.label = column_name
        self.sample_length = sample_length
        self.mileages = defaultdict(list)
        for data in training_set:
            self.mileages[int(data[1]['car'][:-3])].append(data[1][self.label])
        self.max_mileage = {k:max(v) for k,v in self.mileages.items()}
        self.min_mileage = {k:min(v) for k,v in self.mileages.items()}
        self.default_max_mileage = max(self.max_mileage.values())
        self.default_min_mileage = min(self.min_mileage.values())

    def loss(self, batch, mean_pred, is_mse=True, reduce=True):
        label_data = []
        for i in batch[1][self.label]:
            cn = int(batch[1]['car'][0][:-3])
            norm_label = i / 100.0
            label_data.append(norm_label)
        label = torch.tensor(label_data)
        x = mean_pred.squeeze().to("cuda")
        y = label.float().to("cuda")
        mse = torch.nn.MSELoss(reduction='mean')
        loss = 0
        if is_mse:
            if reduce:
                loss = mse(x, y)
            else:
                loss = torch.tensor([mse(x[i], y[i]) for i in range(x.shape[0])])
        return loss


class Task:
    """
    Task
    """

    def __init__(self, columns, encoder_dimension=122, decoder_dimension=122,
                 output_dimension=122, task_name='ev'):
        """
        :param columns: columns
        :param encoder_dimension: encoder dimension int
        :param decoder_dimension: decoder dimension int
        :param output_dimension: output dimension int
        :param task_name: task name, e.g. ev
        """
        self.encoder_dimension = encoder_dimension
        self.decoder_dimension = decoder_dimension
        self.output_dimension = output_dimension
        self.task_name = task_name
        self.columns = columns
        self.encoder = []
        self.decoder = []
        self.target = []
        eval(self.task_name.capitalize() + 'Task.set_params')(self)
        eval(self.task_name.capitalize() + 'Task.get_task_idx')(self, columns)

    def encoder_filter(self, input_embedding):
        return eval(self.task_name.capitalize() + 'Task.task_encoder')(self, input_embedding, self.columns)

    def decoder_filter(self, input_embedding):
        self.decoder = self.encoder[:self.decoder_dimension]
        return to_tensor(to_array(input_embedding)[:, :, self.decoder])

    def target_filter(self, input_embedding):
        self.target = self.encoder[self.decoder_dimension:]
        return to_tensor(to_array(input_embedding)[:, :, self.target])

    def task_encoder(self, input_embedding, columns):
        return to_tensor(to_array(input_embedding)[:, :, self.encoder])


class EvTask(Task):
    def set_params(self):
        """
        initialize
        """
        self.encoder_dimension = 6
        self.decoder_dimension = 2
        self.output_dimension = 4

    def get_task_idx(self, columns):
        """
        filter specified columns
        :param columns: column names
        :Return corresponding indexes
        """
        self.encoder = np.array(
            [columns.index("soc"), columns.index("current"),
             columns.index("max_temp"), columns.index("max_single_volt"),
             columns.index("min_single_volt"), columns.index("volt")]).astype(int)
        return self.encoder


class BatterybrandbTask(Task):
    def set_params(self):
        self.encoder_dimension = 7
        self.decoder_dimension = 4
        self.output_dimension = 3

    def get_task_idx(self, columns):
        self.encoder = np.array(
            [columns.index("soc"),
             columns.index("current"),
             columns.index("min_temp"),
             columns.index("max_single_volt"),
             columns.index("min_single_volt"),
             columns.index('volt'),
             columns.index("max_temp")]).astype(int)
        return self.encoder

class BatterybrandaTask(Task):
    def set_params(self):
        self.encoder_dimension = 7
        self.decoder_dimension = 2
        self.output_dimension = 5

    def get_task_idx(self, columns):
        self.encoder = np.array(
            [columns.index("soc"),
             columns.index("current"),
             columns.index("min_temp"),
             columns.index("max_single_volt"),
             columns.index("max_temp"),
             columns.index("min_single_volt"),
             columns.index('volt')]).astype(int)
        return self.encoder

def to_tensor(input_embedding):
    return to_var(torch.from_numpy(np.array(input_embedding)))


def to_array(input_embedding):
    return input_embedding.cpu().numpy()
