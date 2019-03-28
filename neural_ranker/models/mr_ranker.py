import torch
import numpy as np
from torch.autograd import Variable
from models.base_ranker import BaseRanker


class MRRanker(BaseRanker):
    def __init__(self, epochs, lr):
        """MR ranker without multi-task setting.

        Parameters
        ----------
        epochs : int
            Number of iterations to train.
        lr : float
            Learning rate.
        """

        super().__init__(epochs, lr)

    def train(self, all_features, all_labels):
        """Trains model.

        Parameters
        ----------
        all_features : List
            Train features for individual candidates.
        all_labels : List
            Edit distance labels for individual candidates.
        """

        train_x_1, train_x_2, train_y = self._get_pairwise_features(all_features, all_labels)

        self.set_model(len(all_features[0][0]))
        self.model.training = True

        loss_fn = torch.nn.MarginRankingLoss(margin=1.0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print("Started training.")
        for epoch in range(self.epochs):

            y_pred_1 = self.model(train_x_1)
            y_pred_2 = self.model(train_x_2)
            loss = loss_fn(y_pred_1, y_pred_2, train_y)

            if epoch % 20 == 0:
                 print("Epoch ", epoch, "Loss", loss.data.cpu().numpy().tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Done training.")
        self.model.eval()

    @staticmethod
    def _get_pairwise_features(all_features, all_labels):
        train_labels = []
        train_features_1, train_features_2 = [], []

        for feats, ls in zip(all_features, all_labels):
            for i, sf1 in enumerate(feats):
                for j, sf2 in enumerate(feats):
                    if ls[i] != ls[j]:
                        train_features_1.append(sf1)
                        train_features_2.append(sf2)
                        train_labels.append(float(np.sign(ls[i] - ls[j])))

        train_x_1 = Variable(torch.FloatTensor(train_features_1))
        train_x_2 = Variable(torch.FloatTensor(train_features_2))
        train_y = Variable(torch.FloatTensor(train_labels), requires_grad=False)
        train_y = torch.unsqueeze(train_y, 1)
        return  train_x_1, train_x_2, train_y
