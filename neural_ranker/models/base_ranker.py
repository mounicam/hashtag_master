import torch
import numpy as np
from torch.autograd import Variable


class BaseRanker:
    def __init__(self, epochs, lr):
        """Base class for rankers without multi-task setting.

        Parameters
        ----------
        epochs : int
            Number of iterations to train.
        lr : float
            Learning rate.

        """

        self.epochs = epochs
        self.model = None
        self.lr = lr

    def set_model(self, d_in, dropout=0.5):
        """Initalizes model.

        Parameters
        ----------
        d_in : int
            Input  layer size.
        dropout : float, optional
            Dropout.

        """

        h, d_out = 8, 1
        self.model = torch.nn.Sequential(
            torch.nn.Linear(d_in, h),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(h, h),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(h, h),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(h, h),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(h, d_out)
        )

    def predict(self, test_x):
        """Predicts the relative order of given candidate segmentations pairs.

        Parameters
        ----------
        test_x : List
            List of pariwise feature vectors of candidate pairs.

        Returns
        -------
        List
            List of relative ranking scores.
        """

        test_x = Variable(torch.FloatTensor(test_x))
        return [score[0] for score in self.model(test_x).data.numpy()]
