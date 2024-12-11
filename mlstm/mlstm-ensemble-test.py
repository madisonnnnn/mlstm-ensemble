import csv

import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import auc, recall_score, matthews_corrcoef, roc_curve
from torch import nn

from hdf5 import RNADataset
from mlstm import mLSTM

def calc(TN, FP, FN, TP):
    SN = TP / (TP + FN)  # recall
    SP = TN / (TN + FP)
    Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    return SN, SP, ACC, Precision, F1


class TimeSeriesModel1(torch.nn.Module):
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
class TimeSeriesModel2(torch.nn.Module):
    def __init__(self, input_size=37888, hidden_size=100, n_mlstm=1):
        super().__init__()
        self.mlstm = mLSTM(input_size, hidden_size, n_mlstm)
        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32,2),
            nn.Sigmoid()
        )

    def forward(self, inputs, hidden_states=None):

        _, (h_n, c_n) = self.mlstm(inputs, hidden_states)

        return (h_n, c_n), self.regressor(h_n[-1])
class TimeSeriesModel3(torch.nn.Module):
    def __init__(self, input_size=37888, hidden_size=110, n_mlstm=1):
        super().__init__()
        self.mlstm = mLSTM(input_size, hidden_size, n_mlstm)
        self.regressor = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32,2),
            nn.Sigmoid()
        )

    def forward(self, inputs, hidden_states=None):

        _, (h_n, c_n) = self.mlstm(inputs, hidden_states)

        return (h_n, c_n), self.regressor(h_n[-1])
class TimeSeriesModel4(torch.nn.Module):
    def __init__(self, input_size=37888, hidden_size=110, n_mlstm=1):
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
class TimeSeriesModel5(torch.nn.Module):
    def __init__(self, input_size=37888, hidden_size=120, n_mlstm=1):
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


rng = np.random.RandomState(123)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


num_epochs_LSTM = 5
def mlstm_train(dataset,testset, model, num_epochs,path):
    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(42)
    model = model.to(device)
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128, shuffle=True,drop_last = True)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=128, shuffle=False,drop_last = True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(0, num_epochs):
        hidden_state = None
        # Print epoch
        print(f'Starting epoch {epoch + 1}')
        model.train()
        for i, data in enumerate(trainloader, 0):
            # Get inputs
            inputs, targets = data
            targets = targets.squeeze(1)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.permute(1, 0, 2)
            inputs = inputs.to(torch.float32)

            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            (h_n, c_n), outputs = model(inputs, hidden_state)
            hidden_state = h_n[-1], c_n[-1]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        # Process is complete.
        print('Training process has finished. Saving trained model.')
        torch.save(model, path)
        # Print about testing

    print('Starting testing')
    hidden_state = None
    model.eval()
    prob = []
    prob_all = []
    label_all = []
    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for i, data in enumerate(testloader, 0):
            # Get inputs
            inputs, targets = data
            targets = targets.squeeze(1)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.permute(1, 0, 2)
            inputs = inputs.to(torch.float32)
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Generate outputs
            (h_n, c_n), outputs = model(inputs, hidden_state)
            hidden_state = h_n[-1], c_n[-1]
            _, predicted = torch.max(outputs.data, 1)
            prob_all.extend(predicted.cpu().numpy())
            prob.extend(outputs[:, 1].cpu().numpy())
            label_all.extend(targets.cpu().numpy())

    return prob, prob_all

def pinggu(actuals,predictions,probs):

    roc_auc = roc_auc_score(actuals, predictions)
    tn, fp, fn, tp = confusion_matrix(actuals, probs).ravel()
    sn, sp, acc, precision, f1 = calc(tn, fp, fn, tp)
    fpr, tpr, thresholds = roc_curve(actuals, predictions)
    recall = recall_score(actuals, probs)
    precision1, recall1, thresholds = precision_recall_curve(actuals,predictions)
    auprc = auc(recall1, precision1)
    MCC = matthews_corrcoef(actuals, probs)
    print("*****************************************")
    accuracy = accuracy_score(actuals, probs)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("sn: %.2f%%" % (sn * 100.0))
    print("acc: %.2f%%" % (acc * 100.0))
    print("sp: %.2f%%" % (sp * 100.0))
    print("mcc: %.2f%%" % (MCC * 100.0))
    print("f1: %.2f%%" % (f1 * 100.0))
    print("precision: %.2f%%" % (precision * 100.0))
    print("auc: %.2f%%" % (roc_auc * 100.0))
    data = [acc, MCC, sn, sp, f1, precision, roc_auc, auprc]

    return data,fpr,tpr,thresholds

def avg1(*args):
    result = []
    for items in zip(*args):
        result.append(sum(items) / len(items))
    return result
def avg(p1,p2,p3,p4,p5,a1,a2,a3,a4,a5):

    p = p1.copy()
    p += p2
    p += p3
    p += p4
    p += p5
    a = a1.copy()
    a += a2
    a += a3
    a += a4
    a += a5
    p = (p/5)
    predictions = np.where(a > 2.5, 1, 0)
    return p,predictions










dataset = RNADataset('.h5')
testset = RNADataset('.h5')

f_test = h5py.File('.h5', 'r')
data_cls_test = f_test["labels"]
y_test = data_cls_test[0:].squeeze()


f1 = open('.csv','w', encoding='utf-8', newline="")

csv_write1 = csv.writer(f1, delimiter=',')

model1 = TimeSeriesModel1()
model2 = TimeSeriesModel2()
model3 = TimeSeriesModel3()
model4 = TimeSeriesModel4()
model5 = TimeSeriesModel5()
# model_LSTM = BiLSTM().to(device)

path1 = '.pkl'


predictions1,prob_all1 = mlstm_train(dataset,testset,model1,num_epochs_LSTM,path1)
predictions1 = np.array(predictions1)
# data_best,fp_best, tp_best, th_best = pinggu(y_test, predictions1, prob_all1)
# R_best_roc = open('./result/same/' + cell1[i] + '_1mer_mlstm_test_best_roc' + '.csv', 'w', encoding='utf-8', newline="")
# rows = zip(fp_best, tp_best, th_best)
# title = ['fpr', 'tpr', 't']
# csv_write2 = csv.writer(R_best_roc, delimiter=',')
# csv_write2.writerow(title)
# for row in rows:
#     csv_write2.writerow(row)
# R_best_roc.close()
prob_all1 = np.array(prob_all1)

# predictions7 = train(dataset, model_LSTM, train_index, test_index,num_epochs_LSTM)

predictions2,prob_all2 = mlstm_train(dataset,testset, model2,  num_epochs_LSTM,path2)
predictions2 = np.array(predictions2)
prob_all2 = np.array(prob_all2)

predictions3 ,prob_all3= mlstm_train(dataset,testset, model3, num_epochs_LSTM,path3)
predictions3 = np.array(predictions3)
prob_all3 = np.array(prob_all3)

predictions4,prob_all4 = mlstm_train(dataset, testset,model4, num_epochs_LSTM,path4)
predictions4 = np.array(predictions4)
prob_all4 = np.array(prob_all4)

predictions5,prob_all5 = mlstm_train(dataset, testset,model5, num_epochs_LSTM,path5)
predictions5 = np.array(predictions5)
prob_all5 = np.array(prob_all5)

predictions_same ,pre1= avg(predictions1,predictions2,predictions3,predictions4,predictions5,prob_all1,prob_all2,prob_all3,prob_all4,prob_all5)

data ,fp,tp,th= pinggu (y_test,predictions_same,pre1)

csv_write1.writerow(data)

































