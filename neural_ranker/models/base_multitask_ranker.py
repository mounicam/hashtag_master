import torch
import numpy as np
from abc import abstractmethod
from torch.autograd import Variable


class BaselineMultitaskRanker(torch.nn.Module):
    def __init__(self, epochs, lr_ranker, lr_bin):
        """Base class for rankers with multi-task setting.

        Parameters
        ----------
        epochs : int
            Number of iterations to train.
        lr_ranker : float
            Learning rate for ranker.
        lr_bin : float
            Learning rate for auxilary binary classification task

        """

        super(BaselineMultitaskRanker, self).__init__()
        self.testing = False
        self.epochs = epochs
        self.lr_bin = lr_bin
        self.lr_ranker = lr_ranker

        self.model_bin = None
        self.model_ranker = None
        self.dimen_ranker = None

    def set_model(self, dimen_ranker, dimen_bin):
        """Initialze Model.

        Parameters
        ----------
        dimen_ranker: int
            Input dimensions for ranker.
        dimen_bin: int
            Input dimensions for binary classifier.

        """

        dropout = 0.5
        d_out, h = 1, 8

        print(dimen_ranker, dimen_bin)

        self.model_ranker = torch.nn.Sequential(
            torch.nn.Linear(dimen_ranker, h),
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

        self.model_bin = torch.nn.Sequential(
            torch.nn.Linear(dimen_bin, h),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(h, d_out),
            torch.nn.Sigmoid()
        )

        self.dimen_ranker = dimen_ranker

    def set_testing(self):
        """Set configuration for testing

        """

        self.testing = True
        self.model_ranker.eval()
        self.model_bin.eval()

    def set_training(self):
        """Set configuration for training

        """
        self.model_bin.training = True
        self.model_ranker.training = True

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

        multi, single, hashtag = test_x
        multi = Variable(torch.FloatTensor(multi))
        single = Variable(torch.FloatTensor(single))
        hashtag = Variable(torch.FloatTensor(hashtag))

        _, y_pred = self.forward(multi, single, hashtag)
        return [score[0] for score in y_pred.data.numpy()]

    @abstractmethod
    def forward(self, *input):
        pass
