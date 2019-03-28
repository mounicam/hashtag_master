import torch
import random
import numpy as np
from torch.autograd import Variable
from models.base_multitask_ranker import BaselineMultitaskRanker


class MRMultiRanker(BaselineMultitaskRanker):
    def __init__(self, epochs, lr_ranker, lr_bin):
        """MR ranker with multi-task setting.

        Parameters
        ----------
        epochs : int
            Number of iterations to train.
        lr_ranker : float
            Learning rate for ranker.
        lr_bin : float
            Learning rate for auxilary binary classification task

        """

        super(MRMultiRanker, self).__init__(epochs, lr_ranker, lr_bin)

    def train(self, all_features, all_labels):
        """Trains model.

        Parameters
        ----------
        all_features : List
            Train features for individual candidates.
        all_labels : List
            Edit distance labels for individual candidates.
        """

        multi_x, single_x, hashtag_x, train_y, train_bin_y = self._get_pairwise_features(all_features, all_labels)

        self.set_model(len(all_features[0][0][0]), len(all_features[0][0][2]))
        self.set_training()

        loss_fn_bin = torch.nn.BCELoss()
        loss_fn_ranker = torch.nn.MarginRankingLoss(margin=1.0)
        optimizer_bin = torch.optim.Adam(self.model_bin.parameters(), lr=self.lr_bin)
        optimizer_ranker = torch.optim.Adam(self.model_ranker.parameters(), lr=self.lr_ranker)

        print("Started training.")
        for epoch in range(self.epochs):

            y_pred_bin, y_pred_ranker_1, y_pred_ranker_2 = self.forward(multi_x, single_x, hashtag_x)

            loss1 = loss_fn_ranker(y_pred_ranker_1, y_pred_ranker_2, train_y)
            loss2 = loss_fn_bin(y_pred_bin, train_bin_y)
            loss = loss1 + loss2

            if epoch % 20 == 0:
                 print("Epoch ", epoch, "Loss", loss.data.cpu().numpy().tolist())

            optimizer_bin.zero_grad()
            optimizer_ranker.zero_grad()
            loss.backward()
            optimizer_bin.step()
            optimizer_ranker.step()

        self.set_testing()
        print("Done training.")

    def forward(self, multi_x, single_x, hashtag_x):
        """Trains model.

        Parameters
        ----------
        multi_x : List
            List of pairwise feature vectors that work well for multi-word hashtags.
        single_x : List
            List of pairwise feature vectors that work well for single-word hashtags.
        hashtag_x : List
            List of feature vectors for hashtags.
        """

        gate = self.model_bin(hashtag_x)
        gate_exp = gate.expand(list(gate.shape)[0], self.dimen_ranker)

        if self.testing:
            lm_tensor = (1 - gate_exp) * single_x + gate_exp * multi_x
            return gate, self.model_ranker(lm_tensor)
        else:
            lm_tensor_1 = (1 - gate_exp) * single_x[0] + gate_exp * multi_x[0]
            lm_tensor_2 = (1 - gate_exp) * single_x[1] + gate_exp * multi_x[1]
            return gate, self.model_ranker(lm_tensor_1), self.model_ranker(lm_tensor_2)

    @staticmethod
    def _get_pairwise_features(all_features, all_labels):
        multi_x_1, multi_x_2, single_x_1, single_x_2, hashtag_x, ediff, bin = [], [], [], [], [], [], []

        for feats, ls in zip(all_features, all_labels):
            for i, s1 in enumerate(feats):
                for j, s2 in enumerate(feats):
                    if ls[i][0] != ls[j][0]:
                        multi_x_1.append(s1[0])
                        multi_x_2.append(s2[0])
                        single_x_1.append(s1[1])
                        single_x_2.append(s2[1])
                        hashtag_x.append(s1[2])
                        ediff.append(float(np.sign(ls[i][0] - ls[j][0])))
                        bin.append(ls[i][1])

        multi_x = (Variable(torch.FloatTensor(multi_x_1)), Variable(torch.FloatTensor(multi_x_2)))
        single_x = (Variable(torch.FloatTensor(single_x_1)), Variable(torch.FloatTensor(single_x_2)))
        hashtag_x = Variable(torch.FloatTensor(hashtag_x))
        train_y = Variable(torch.FloatTensor(ediff), requires_grad=False)
        train_bin_y = Variable(torch.FloatTensor(bin), requires_grad=False)
        train_y = torch.unsqueeze(train_y, 1)
        train_bin_y = torch.unsqueeze(train_bin_y, 1)
        return multi_x, single_x, hashtag_x, train_y, train_bin_y
