from pickle import load
from tensorflow.keras.metrics import (FalseNegatives, FalsePositives,
                                      TrueNegatives, TruePositives,
                                      Recall, Precision)
from torch.nn import Sequential, Linear, Sigmoid, ReLU, Module, BatchNorm1d, Dropout
from torch import save
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from clearml import Task, Logger
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import torch.nn.functional as F

# metrics
def f1(t, p):
    tp = np.logical_and(t, p).astype(float).sum()
    fp = np.less(t, p).astype(float).sum()
    fn = np.less(p, t).astype(float).sum()
    prec = tp/(tp+fp) #precision
    rec = tp/(tp+fn)  #recall
    return 2*((prec*rec)/(prec+rec))


def accuracy(pred, target):
    return len(np.nonzero(pred == target)[0]) / (pred.shape[0] * pred.shape[1])

#loader
class MoleculeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#model
class Chem(Module):
    def __init__(self, in_f, n=1000):
        super().__init__()
        self.in_f = in_f

        self.model_body = Sequential(
            Linear(in_f, n),  # 1 2 3 4
            # BatchNorm1d(n),  # 3 4
            ReLU(inplace=True),
            # Dropout(p=0.3, inplace=False),  # 4
            # Linear(n, 500),  # 4
            # BatchNorm1d(500),  # 4
            # ReLU(inplace=True),  # 4
            # Linear(500, 1)  # 4
            Linear(n, 1)  # 2 3
        )
        self.head = Sigmoid()

    def forward(self, x):
        x = self.model_body(x)
        y = self.head(x)
        return y


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task = Task.init(project_name="filter_model", task_name="NN with two hidden layears batch norm and dropout")
    with open('./fps_pkls/united_lfp_6_4096_230322.pickle', 'rb') as f:
        data = load(f)

    X = np.stack([l[0][0] for l in list(data.values())])
    Y = np.array([l[1] for l in list(data.values())])
    Y = Y.astype('float32', copy=False)

    x_pretrain, x_test, y_pretrain, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    x_train, x_valid, y_train, y_valid = train_test_split(x_pretrain,
                                                          y_pretrain,
                                                          test_size=0.2,
                                                          random_state=42,
                                                          stratify=y_pretrain)

    train_ds = MoleculeDataset(x_train, y_train)
    valid_ds = MoleculeDataset(x_valid, y_valid)

    train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=1024, shuffle=False, drop_last=True)

    in_f = x_train.shape[1]
    model = Chem(in_f)

    epochs = 50
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss(reduction='mean')

    threshold = 0.5
    log_freq = 100
    model.to(device)

    # train
    for epoch in range(epochs):
        for batch_ind, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)

            x = x.to(torch.float32)
            y = y.reshape((y.shape[0], 1))

            y_pred = model(x)

            loss = criterion(y_pred, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if batch_ind % log_freq == 0 and batch_ind > 0:
                Logger.current_logger().report_scalar(
                    "train", "loss", iteration=(epoch * len(train_dl) + batch_ind), value=loss.item())
                # посчитать метрики на traininig set ACC, BA, F1
                print('[%d/%d][%d/%d] Loss: %.4f ' % (epoch, epochs, batch_ind,
                                                      len(train_dl),
                                                      loss.item()))

        # Evaluation
        with torch.no_grad():
            avg_acc = 0
            avg_f1 = 0
            avg_loss = 0
            avg_balanced_accuracy_score = 0

            for x, y in valid_dl:
                x, y = x.to(device), y.to(device)

                x = x.to(torch.float32)
                y = y.reshape((y.shape[0], 1))

                y_pred = model(x)

                loss = criterion(y_pred, y)
                avg_loss += loss

                pred_transformed = y_pred > threshold
                m = accuracy(pred_transformed.cpu().numpy(), y.cpu().numpy())
                avg_acc += m
                f = f1(y.cpu().numpy(), pred_transformed.cpu().numpy())
                avg_f1 += f

                ba = balanced_accuracy_score(y.cpu().numpy(), pred_transformed.cpu().numpy())
                avg_balanced_accuracy_score += ba

            avg_acc = avg_acc / len(valid_dl)
            avg_loss = avg_loss / len(valid_dl)
            avg_f1 = avg_f1 / len(valid_dl)
            avg_ba = avg_balanced_accuracy_score / len(valid_dl)

            Logger.current_logger().report_scalar(
                "valid", "loss", iteration=(epoch * len(valid_dl) + batch_ind), value=loss.item())
            print(
                '[{:d}/{:d}] Accuracy {:.3f} F1-score {:.3f}  BA {:.3f}| Loss: {:.4f}\n'.format(epoch, epochs, avg_acc,
                                                                                                avg_f1, avg_ba,
                                                                                                avg_loss.item()))

    test_ds = MoleculeDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=1024, shuffle=False, drop_last=True)

    # test
    # model.to(device)
    predictions = []
    target = []

    with torch.no_grad():
        avg_m, avg_f1, avg_balanced_accuracy_score = 0, 0, 0
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)

            x = x.to(torch.float32)
            y = y.reshape((y.shape[0], 1))

            target.append(y.cpu().numpy())
            pred = model(x)
            predictions.append(pred)
            pred_transformed = pred > threshold
            m = accuracy(pred_transformed.cpu().numpy(), y.cpu().numpy())  #
            avg_m += m
            f = f1(y.cpu().numpy(), pred_transformed.cpu().numpy())
            avg_f1 += f

            ba = balanced_accuracy_score(y.cpu().numpy(), pred_transformed.cpu().numpy())
            avg_balanced_accuracy_score += ba

        avg_m = avg_m / len(test_dl)
        avg_f1 = avg_f1 / len(test_dl)
        avg_ba = avg_balanced_accuracy_score / len(test_dl)
        print('Accuracy {:.3f} F1-score {:.3f}  BA {:.3f}\n'.format(avg_m, avg_f1, avg_ba))