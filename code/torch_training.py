from pickle import load
from torch.nn import Sequential, Linear, Sigmoid, ReLU, Module, BatchNorm1d, Dropout
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from clearml import Task, Logger
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch


# metrics
def f1(t, p):
    tp = np.logical_and(t, p).astype(float).sum()
    fp = np.less(t, p).astype(float).sum()
    fn = np.less(p, t).astype(float).sum()
    prec = tp / (tp + fp)  # precision
    rec = tp / (tp + fn)  # recall
    return 2 * ((prec * rec) / (prec + rec))


def accuracy(pred, target):
    return len(np.nonzero(pred == target)[0]) / (pred.shape[0] * pred.shape[1])


# loader
class MoleculeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# model
class Chem(Module):
    def __init__(self, in_f, n=1000, k=500):
        super().__init__()
        self.in_f = in_f

        self.model_body = Sequential(
            Linear(in_f, n),  # 1 2 3 4 5 6
            ReLU(inplace=True),  # 2 3 4 5 6
            # BatchNormalization(trainable=True),  # 3  НЕ РАБОТАЕТ
            # BatchNorm1d(n),  # 3
            # Linear(n, 1),  # 2 3
            Dropout(p=0.3, inplace=False),  # 6
            Linear(n, k),  # 4 5 6
            ReLU(inplace=True),  # 4 5 6
            BatchNorm1d(k),  # 5 6
            Linear(k, 1)  # 4 5 6
        )
        self.head = Sigmoid()

    def forward(self, x):
        x = self.model_body(x)
        y = self.head(x)
        return y


# logging
def report_scalars_on_batch(*args):
    avg_acc, avg_loss, avg_f1, avg_ba, sample, epoch, dl, batch_ind = args
    for name, metric in [avg_acc, avg_loss, avg_f1, avg_ba]:
        Logger.current_logger().report_scalar(
                sample, name, iteration=(epoch * len(dl) + batch_ind), value=metric)


def report_scalars(*args):
    avg_acc, avg_loss, avg_f1, avg_ba, sample, epoch = args
    for name, metric in [avg_acc, avg_loss, avg_f1, avg_ba]:
        if all([x is not None for x in [name, metric]]):
            Logger.current_logger().report_scalar(
                sample, name, iteration=epoch, value=metric)


def report_test_plots(avg_m, avg_f1, avg_ba, model_name, predictions, target, series):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import confusion_matrix
    # reporting pd table
    df = pd.DataFrame(
        {
            "acc": [avg_m],
            "f1": [avg_f1],
            "bac": [avg_ba],
        },
        index=[model_name],
    )
    df.index.name = "model_name"
    Logger.current_logger().report_table(
        "table pd",
        model_name,
        iteration=0,
        table_plot=df
    )

    # reporting confusion matrix
    labels = ["Reconstructed", "Decoy"]
    cm = confusion_matrix(target, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)

    Logger.current_logger().report_matplotlib_figure(
        title="Confusion matrix",
        series=series,
        iteration=0,
        figure=plt,
    )


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "logistic regression"
    task = Task.init(project_name="filter_model", task_name=model_name)
    with open("./fps_pkls/united_lfp_6_4096_230322.pickle", 'rb') as f:
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
    Logger.current_logger().report_text(str(model))

    epochs = 100
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss(reduction='mean')

    threshold = 0.5
    log_freq = 100
    model.to(device)

    for epoch in range(1, epochs + 1):

        train_avg_acc, train_avg_bal_acc, train_avg_f1, train_avg_loss = 0, 0, 0, 0
        for batch_ind, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)

            x = x.to(torch.float32)
            y = y.reshape((y.shape[0], 1))

            y_pred = model(x)

            loss = criterion(y_pred, y)
            train_avg_loss += loss

            pred_transformed = y_pred > threshold
            m = accuracy(pred_transformed.cpu().numpy(), y.cpu().numpy())
            train_avg_acc += m

            f = f1(y.cpu().numpy(), pred_transformed.cpu().numpy())
            train_avg_f1 += f

            ba = balanced_accuracy_score(y.cpu().numpy(), pred_transformed.cpu().numpy())
            train_avg_bal_acc += ba

            loss.backward()
            opt.step()
            opt.zero_grad()
            if batch_ind % log_freq == 0 and batch_ind > 0:
                report_scalars_on_batch(("acc", train_avg_acc / batch_ind),
                                        ("loss", train_avg_loss.item() / batch_ind),
                                        ("f1", train_avg_f1 / batch_ind),
                                        ("bac", train_avg_bal_acc / batch_ind),
                                        "train_batch", epoch, train_dl, batch_ind)
                print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, epochs, batch_ind,
                                                     len(train_dl),
                                                     loss.item()))

        train_avg_acc /= len(train_dl)
        train_avg_loss /= len(train_dl)
        train_avg_f1 /= len(train_dl)
        train_avg_bal_acc /= len(train_dl)

        # reporting metric scalars on train set
        report_scalars(("acc_train", train_avg_acc),
                       ("loss_train", train_avg_loss.item()),
                       ("f1_train", train_avg_f1),
                       ("bac_train", train_avg_bal_acc),
                       "train_epoch", epoch)
        # reporting all metrics into one plot
        report_scalars(("acc_train", train_avg_acc),
                       ("loss_train", train_avg_loss.item()),
                       ("f1_train", train_avg_f1),
                       ("bac_train", train_avg_bal_acc),
                       "train_with_validation_epoch", epoch)

        # Evaluation
        with torch.no_grad():

            valid_avg_acc, valid_avg_f1, valid_avg_loss, valid_avg_bal_acc = 0, 0, 0, 0
            for x, y in valid_dl:
                x, y = x.to(device), y.to(device)

                x = x.to(torch.float32)
                y = y.reshape((y.shape[0], 1))

                y_pred = model(x)

                loss = criterion(y_pred, y)
                valid_avg_loss += loss

                pred_transformed = y_pred > threshold
                m = accuracy(pred_transformed.cpu().numpy(), y.cpu().numpy())
                valid_avg_acc += m

                f = f1(y.cpu().numpy(), pred_transformed.cpu().numpy())
                valid_avg_f1 += f

                ba = balanced_accuracy_score(y.cpu().numpy(), pred_transformed.cpu().numpy())
                valid_avg_bal_acc += ba

            valid_avg_acc /= len(valid_dl)
            valid_avg_loss /= len(valid_dl)
            valid_avg_f1 /= len(valid_dl)
            valid_avg_bal_acc /= len(valid_dl)

            # reporting metric scalars on validation set
            report_scalars(("acc_valid", valid_avg_acc),
                           ("loss_valid", valid_avg_loss.item()),
                           ("f1_valid", valid_avg_f1),
                           ("bac_valid", valid_avg_bal_acc),
                           "validation_epoch", epoch)
            # reporting loss on test and validation set
            report_scalars((None, None), (None, None),
                           ("loss_train", train_avg_loss.item()),
                           ("loss_valid", valid_avg_loss.item()),
                           "train_and_validation_loss", epoch)
            # merge all metrics into one plot
            report_scalars(("acc_valid", valid_avg_acc),
                           ("loss_valid", valid_avg_loss.item()),
                           ("f1_valid", valid_avg_f1),
                           ("bac_valid", valid_avg_bal_acc),
                           "train_with_validation_epoch", epoch)
            print('[{:d}/{:d}] Accuracy {:.3f} F1-score {:.3f}  BA {:.3f}| Loss: {:.4f}\n'.format(epoch, epochs,
                                                                                                  valid_avg_acc,
                                                                                                  valid_avg_f1,
                                                                                                  valid_avg_bal_acc,
                                                                                                  valid_avg_loss.item()))

    test_ds = MoleculeDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=1024, shuffle=False, drop_last=True)

    # test
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

            m = accuracy(pred_transformed.cpu().numpy(), y.cpu().numpy())
            avg_m += m

            f = f1(y.cpu().numpy(), pred_transformed.cpu().numpy())
            avg_f1 += f

            ba = balanced_accuracy_score(y.cpu().numpy(), pred_transformed.cpu().numpy())
            avg_balanced_accuracy_score += ba

        avg_m = avg_m / len(test_dl)
        avg_f1 = avg_f1 / len(test_dl)
        avg_ba = avg_balanced_accuracy_score / len(test_dl)

        report_test_plots(avg_m, avg_f1, avg_ba, model_name,
                          np.array([int(x.item()) for x in pred_transformed]),
                          np.array([int(x.item()) for x in target[-1]]), "On last batch")

        report_test_plots(avg_m, avg_f1, avg_ba, model_name,
                          np.array([1 if x.item() > threshold else 0 for tens in predictions for x in tens]),
                          np.array([int(x.item()) for arr in target for x in arr]), "All")

        print('Accuracy {:.3f} F1-score {:.3f}  BA {:.3f}\n'.format(avg_m, avg_f1, avg_ba))
