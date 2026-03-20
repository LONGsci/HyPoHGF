import sys
import os
import numpy as np
import torch
import time
from pytorchtools import EarlyStopping
from models import HyPoHGF
from config import params, checkpoint_dir, seed
from utils import Logger
from data_processing import (sa_data, ss_data, aa_x1, aa_x2, aa_edge_index, aa_edge_index2,
                             sa_weights, ss_weights, aa_weights, aa_weights2,
                             train_loader, dev_loader,test_loader,x_dev,x_test)

np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

timestamp = time.strftime("%Y%m%d-%H%M")
sys.stdout = Logger(f'{checkpoint_dir}/HyPoHGF_{timestamp}.txt')

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
print("lr: ", params.lr, " rec: ", params.rec, " dropout: ", params.drop, " batchsize: ",
      params.batchSize, " epoch: ", params.epoch, " dev_ratio: ", params.dev_ratio, " test_ratio: ", params.test_ratio)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = HyPoHGF(906, 512, 256, params.batchSize, params.drop)

optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.rec)

criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

early_stopping = EarlyStopping(patience=41, verbose=True)
print('device: ', device)

epsilon = 1e-13

for epoch in range(params.epoch):

    model.train()
    running_loss = 0.0
    for i, (sid, hid) in enumerate(train_loader):

        sid, hid = sid.float(), hid.float()
        optimizer.zero_grad()

        outputs = model(sa_data.x, sa_data.edge_index, sa_weights, ss_data.x, ss_data.edge_index, ss_weights,
                        aa_x1, aa_x2, aa_edge_index, aa_edge_index2, aa_weights, aa_weights2, sid)

        loss = criterion(outputs, hid)

        loss.backward()
        if i % 5 == 0:
            total_grad_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm

        optimizer.step()
        running_loss += loss.item()
    print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))

    model.eval()
    dev_loss = 0

    dev_p2 = 0
    dev_p5 = 0
    dev_p10 = 0

    dev_r2 = 0
    dev_r5 = 0
    dev_r10 = 0

    dev_f1_2 = 0
    dev_f1_5 = 0
    dev_f1_10 = 0

    for tsid, taid in dev_loader:
        tsid, taid = tsid.float(), taid.float()

        outputs = model(sa_data.x, sa_data.edge_index, sa_weights, ss_data.x, ss_data.edge_index, ss_weights,
                        aa_x1, aa_x2, aa_edge_index, aa_edge_index2, aa_weights, aa_weights2, tsid)
        dev_loss += criterion(outputs, taid).item()

        for i, hid in enumerate(taid):
            trueLabel = []
            for idx, val in enumerate(hid):
                if val == 1:
                    trueLabel.append(idx)

            top2 = torch.topk(outputs[i], 2)[1]
            count = 0
            for m in top2:
                if m in trueLabel:
                    count += 1
            dev_p2 += count / 2
            dev_r2 += count / len(trueLabel)

            top5 = torch.topk(outputs[i], 5)[1]
            count = 0
            for m in top5:
                if m in trueLabel:
                    count += 1
            dev_p5 += count / 5
            dev_r5 += count / len(trueLabel)

            top10 = torch.topk(outputs[i], 10)[1]
            count = 0
            for m in top10:
                if m in trueLabel:
                    count += 1
            dev_p10 += count / 10
            dev_r10 += count / len(trueLabel)

    scheduler.step()

    print('[Epoch {}]dev_loss: '.format(epoch + 1), dev_loss / len(dev_loader))

    print('p2-5-10:', dev_p2 / len(x_dev), dev_p5 / len(x_dev), dev_p10 / len(x_dev))
    print('r2-5-10:', dev_r2 / len(x_dev), dev_r5 / len(x_dev), dev_r10 / len(x_dev))

    print('f1_2-5-10: ',
          2 * (dev_p2 / len(x_dev)) * (dev_r2 / len(x_dev)) / ((dev_p2 / len(x_dev)) + (dev_r2 / len(x_dev)) + epsilon),
          2 * (dev_p5 / len(x_dev)) * (dev_r5 / len(x_dev)) / ((dev_p5 / len(x_dev)) + (dev_r5 / len(x_dev)) + epsilon),
          2 * (dev_p10 / len(x_dev)) * (dev_r10 / len(x_dev)) / (
                      (dev_p10 / len(x_dev)) + (dev_r10 / len(x_dev)) + epsilon))

    early_stopping(dev_loss / len(dev_loader), model)
    if early_stopping.early_stop:
        print("Early stopping")
        current_date = time.strftime("%Y-%m-%d-%H_%M")
        checkpoint_filename = f'{checkpoint_dir}/checkpoint_HyPoHGF_{current_date}.pt'
        torch.save(model.state_dict(), checkpoint_filename)
        break

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

model.load_state_dict(torch.load(checkpoint_filename))

model.eval()
test_loss = 0

test_p2 = 0
test_p5 = 0
test_p10 = 0

test_r2 = 0
test_r5 = 0
test_r10 = 0

test_f1_2 = 0
test_f1_5 = 0
test_f1_10 = 0

for tsid, taid in test_loader:
    tsid, taid = tsid.float(), taid.float()

    outputs = model(sa_data.x, sa_data.edge_index, sa_weights, ss_data.x, ss_data.edge_index, ss_weights,
                    aa_x1, aa_x2, aa_edge_index, aa_edge_index2, aa_weights, aa_weights2, tsid)
    test_loss += criterion(outputs, taid).item()

    for i, hid in enumerate(taid):
        trueLabel = []
        for idx, val in enumerate(hid):
            if val == 1:
                trueLabel.append(idx)

        top3 = torch.topk(outputs[i], 2)[1]
        count = 0
        for m in top3:
            if m in trueLabel:
                count += 1
        test_p2 += count / 2
        test_r2 += count / len(trueLabel)

        top5 = torch.topk(outputs[i], 5)[1]
        count = 0
        for m in top5:
            if m in trueLabel:
                count += 1
        test_p5 += count / 5
        test_r5 += count / len(trueLabel)

        top10 = torch.topk(outputs[i], 10)[1]
        count = 0
        for m in top10:
            if m in trueLabel:
                count += 1
        test_p10 += count / 10
        test_r10 += count / len(trueLabel)

print("----------------------------------------------------------------------------------------------------")

print('test_loss: ', test_loss / len(test_loader))
print('p2-5-10:', test_p2 / len(x_test), test_p5 / len(x_test), test_p10 / len(x_test))
print('r2-5-10:', test_r2 / len(x_test), test_r5 / len(x_test), test_r10 / len(x_test))

print('f1_2-5-10: ',
      2 * (test_p2 / len(x_test)) * (test_r2 / len(x_test)) / ((test_p2 / len(x_test)) + (test_r2 / len(x_test))),
      2 * (test_p5 / len(x_test)) * (test_r5 / len(x_test)) / ((test_p5 / len(x_test)) + (test_r5 / len(x_test))),
      2 * (test_p10 / len(x_test)) * (test_r10 / len(x_test)) / ((test_p10 / len(x_test)) + (test_r10 / len(x_test))))
