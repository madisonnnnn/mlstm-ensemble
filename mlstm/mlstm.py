import csv

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_recall_curve, auc, \
    matthews_corrcoef, roc_curve
from torch import nn

from hdf5 import RNADataset
from mlstm import mLSTM

eval_losses = []
eval_acces = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
def calc(TN, FP, FN, TP):
    SN = TP / (TP + FN)  # recall
    SP = TN / (TN + FP)
    Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    return SN, SP, ACC, Precision, F1
class TimeSeriesModel(torch.nn.Module):
    def __init__(self, input_size=37888, hidden_size=100, n_mlstm=1):
        super().__init__()
        self.mlstm = mLSTM(input_size, hidden_size, n_mlstm)
        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16,2),
            nn.Sigmoid()
        )

    def forward(self, inputs, hidden_states=None):

        _, (h_n, c_n) = self.mlstm(inputs, hidden_states)

        return (h_n, c_n), self.regressor(h_n[-1])


# criterion = torch.nn.HuberLoss()
criterion = nn.CrossEntropyLoss()
dataset1 = RNADataset('./h.B/5mer/h_B_all.Train.label.h5')
train_db, val_db = torch.utils.data.random_split(dataset1, [7368, 1842])
testset = RNADataset('./h.B/5mer/h_B_all.Test.label.h5')

torch.manual_seed(42)
R =open('./调试/h_b_5mer.csv', 'w', encoding='utf-8',newline="")
csv_write = csv.writer(R, delimiter=',')
titel = ['epoch', 'batch','lr','train_loss', 'val_loss','val_acc', 'val_auc', 'test_loss','test_acc', 'test_auc']
csv_write.writerow(titel)
    # Print

batch=[32,64,128]
learn_rate=[0.0001,0.00001,0.001]

for bz in range(3):
    for learn in range(3):
        trainloader1 = torch.utils.data.DataLoader(train_db, batch_size=batch[bz],drop_last = True,shuffle=True)
        testloader1 = torch.utils.data.DataLoader(val_db, batch_size=batch[bz],drop_last = True,shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch[bz], drop_last=True,shuffle=False)
        model = TimeSeriesModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), learn_rate[learn])

        for epoch in range(100):
            print(f'Starting epoch {epoch + 1}')
            losses_train, losses_valid = [], []
            train_loss=0
            hidden_state = None
            model.train()
            for inputs, targets in trainloader1:
                inputs = inputs.unsqueeze(1)
                inputs = inputs.permute(1,0,2)

                targets = targets.squeeze(1)
                # n_batch, n_feat = inputs.size()
                # inputs = inputs.reshape(n_batch,41,4)
                # cnn输入
                inputs = inputs.to(torch.float32)
                inputs = inputs.cuda()
                targets = targets.cuda()

                (h_n, c_n), predictions = model(inputs, hidden_state)

                hidden_state = h_n[-1], c_n[-1]
                loss = criterion(predictions, targets)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()
            loss_train =train_loss / len(trainloader1)

            hidden_state = None
            model.eval()
            correct, total = 0, 0
            eval_loss = 0
            prob = []
            prob_all = []
            label_all = []
            for inputs, targets in testloader1:
                inputs = inputs.unsqueeze(1)
                inputs = inputs.permute(1,0,2)
                targets = targets.squeeze(1)
                # cnn输入
                inputs = inputs.to(torch.float32)
                inputs = inputs.cuda()
                targets = targets.cuda()

                with torch.no_grad():
                    (h_n, c_n), outputs = model(inputs, hidden_state)
                    hidden_state = h_n[-1], c_n[-1]
                    loss = criterion(outputs, targets)
                    eval_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                prob.extend(outputs[:, 1].cpu().numpy())
                prob_all.extend(predicted.cpu().numpy())
                label_all.extend(targets.cpu().numpy())
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            loss_eval=(eval_loss / len(testloader1))
            tn, fp, fn, tp = confusion_matrix(label_all, prob_all).ravel()
            sn, sp, val_acc, precision, f1 = calc(tn, fp, fn, tp)
            fpr, tpr, thresholds = roc_curve(label_all, prob)
            # csv_write1.writerow(tpr)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

            recall = recall_score(label_all, prob_all)
            precision1, recall1, thresholds = precision_recall_curve(label_all, prob)
            auprc = auc(recall1, precision1)
            val_roc_auc = roc_auc_score(label_all, prob)
            MCC = matthews_corrcoef(label_all, prob_all)
            print('Accuracy for fold : %d %%' % ( 100.0 * correct / total))
            print("sn: %.2f%%" % (sn * 100.0))
            print("sp: %.2f%%" % (sp * 100.0))
            print("mcc: %.2f%%" % (MCC * 100.0))
            print("f1: %.2f%%" % (f1 * 100.0))
            print("precision: %.2f%%" % (precision * 100.0))
            print("auprc:{:.4f}".format(auprc))
            print('--------------------------------')


            hidden_state = None
            model.eval()
            correct, total = 0, 0
            test_loss = 0
            prob = []
            prob_all = []
            label_all = []
            for inputs, targets in testloader:
                inputs = inputs.unsqueeze(1)
                inputs = inputs.permute(1, 0, 2)
                targets = targets.squeeze(1)
                # cnn输入
                inputs = inputs.to(torch.float32)
                inputs = inputs.cuda()
                targets = targets.cuda()

                with torch.no_grad():
                    (h_n, c_n), outputs = model(inputs, hidden_state)
                    hidden_state = h_n[-1], c_n[-1]
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                prob.extend(outputs[:, 1].cpu().numpy())
                prob_all.extend(predicted.cpu().numpy())
                label_all.extend(targets.cpu().numpy())
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            loss_test = (test_loss / len(testloader))
            tn, fp, fn, tp = confusion_matrix(label_all, prob_all).ravel()
            sn, sp, test_acc, precision, f1 = calc(tn, fp, fn, tp)
            fpr, tpr, thresholds = roc_curve(label_all, prob)
            # csv_write1.writerow(tpr)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

            recall = recall_score(label_all, prob_all)
            precision1, recall1, thresholds = precision_recall_curve(label_all, prob)
            auprc = auc(recall1, precision1)
            test_roc_auc = roc_auc_score(label_all, prob)
            MCC = matthews_corrcoef(label_all, prob_all)
            print('Accuracy for fold : %d %%' % ( 100.0 * correct / total))
            print("sn: %.2f%%" % (sn * 100.0))

            print("sp: %.2f%%" % (sp * 100.0))
            print("mcc: %.2f%%" % (MCC * 100.0))
            print("f1: %.2f%%" % (f1 * 100.0))
            print("precision: %.2f%%" % (precision * 100.0))

            print("auprc:{:.4f}".format(auprc))
            print('--------------------------------')
            data = [epoch,batch[bz],learn_rate[learn],loss_train,loss_eval,val_acc,val_roc_auc,loss_test,test_acc, test_roc_auc]

            print(data)


            csv_write.writerow(data)


