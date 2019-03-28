import torch
from models.base_ranker import BaseRanker
from torch.autograd import Variable


class MSERanker(BaseRanker):
    def __init__(self, epochs, lr):
        """MSE ranker without multi-task setting.

        Parameters
        ----------
        epochs : int
            Number of iterations to train.
        lr : float
            Learning rate.
        """

        super(MSERanker, self).__init__(epochs, lr)

    def train(self, all_features, all_labels):
        """Trains model.

        Parameters
        ----------
        all_features : List
            Train features for individual candidates.
        all_labels : List
            Edit distance labels for individual candidates.
        """

        train_x, train_y = self._get_pairwise_features(all_features, all_labels)

        self.set_model(len(all_features[0][0]) * 2)
        self.model.training = True

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print("Started training.")
        for epoch in range(self.epochs):

            y_pred = self.model(train_x)
            loss = loss_fn(y_pred, train_y)

            if epoch % 20 == 0:
                 print("Epoch", epoch, "Loss", loss.data.cpu().numpy().tolist())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Done training.")
        self.model.eval()

    @staticmethod
    def _get_pairwise_features(all_features, all_labels):
        train_features = []
        train_labels = []

        for feats, ls in zip(all_features, all_labels):
            for i, sf1 in enumerate(feats):
                for j, sf2 in enumerate(feats):
                    train_features.append(sf1 + sf2)
                    train_labels.append(ls[i] - ls[j])

        train_x = Variable(torch.FloatTensor(train_features))
        train_y = Variable(torch.FloatTensor(train_labels), requires_grad=False)
        train_y = torch.unsqueeze(train_y, 1)
        return train_x, train_y
