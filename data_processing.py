import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from utils import presDataset
from config import params

sa_edge = np.load('./data/SA-edges.npy')
sa_weights = np.load('./data/SA-weights.npy')
sa_weights = torch.tensor(sa_weights, dtype=torch.float32)
sa_weights = torch.log(sa_weights + 1)
sa_weights = (sa_weights - sa_weights.min()) / (sa_weights.max() - sa_weights.min())
sa_weights = sa_weights * 0.99 + 0.01
sa_edge = sa_edge.tolist()
sa_edge_index = torch.tensor(sa_edge, dtype=torch.long).t()
sa_x = torch.tensor([i for i in range(880)], dtype=torch.float32)
sa_data = Data(x=sa_x, edge_index=sa_edge_index.contiguous())


ss_edge = np.load('./data/symptom_edges.npy')
ss_weights = np.load('./data/symptom_edges_weights.npy').squeeze()
ss_weights = torch.tensor(ss_weights, dtype=torch.float32)
ss_weights = torch.log(ss_weights + 1)
ss_weights = (ss_weights - ss_weights.min()) / (ss_weights.max() - ss_weights.min())
ss_weights = ss_weights * 0.99 + 0.01
ss_edge = ss_edge.tolist()
ss_edge_index = torch.tensor(ss_edge, dtype=torch.long).t()
ss_x = torch.tensor([i for i in range(466)], dtype=torch.float32)
ss_data = Data(x=ss_x, edge_index=ss_edge_index.contiguous())


aa_edge = np.load('./data/acupoint_edges.npy').tolist()
aa_edge2 = np.load('./data/acuhetnet.npy').tolist()
aa_weights = np.load('./data/acupoint_edges_weights.npy').squeeze()
aa_weights = torch.tensor(aa_weights, dtype=torch.float32)
aa_weights = torch.log(aa_weights + 1)
aa_weights = (aa_weights - aa_weights.min()) / (aa_weights.max() - aa_weights.min())
aa_weights = aa_weights * 0.99 + 0.01
aa_weights2 = torch.cat([
    torch.full((338,), 0.5, dtype=torch.float32),
    torch.full((127,), 0.1, dtype=torch.float32),
    torch.full((338,), 0.5, dtype=torch.float32),
    torch.full((127,), 0.1, dtype=torch.float32)
])
aa_edge_index = torch.tensor(aa_edge, dtype=torch.long) - 466
aa_edge_index = aa_edge_index.t().contiguous()
aa_edge_index2 = torch.tensor(aa_edge2, dtype=torch.long) - 466
aa_edge_index2=aa_edge_index2.t().contiguous()
aa_x1 = torch.tensor([i for i in range(466, 880 )], dtype=torch.float)
aa_x2 = torch.tensor([i for i in range(880, 880 + 26)], dtype=torch.float)


prescript = pd.read_csv('./data/prescript.csv', encoding='utf-8-sig')
pLen = len(prescript)

pS_array = torch.zeros(pLen, 466, dtype=torch.float32)

pH_array = torch.zeros(pLen, 414, dtype=torch.float32)
for i in range(pLen):
    j = eval(prescript.iloc[i, 0])
    pS_array[i, j] = 1
    k = eval(prescript.iloc[i, 1])
    k = [x - 466 for x in k]
    pH_array[i, k] = 1


p_list = [x for x in range(pLen)]
x_train, x_dev_test = train_test_split(p_list, test_size= (params.dev_ratio+params.test_ratio), shuffle=True, random_state=2021)
x_dev, x_test = train_test_split(x_dev_test, test_size=1 - 0.5, shuffle=True, random_state=2021)

train_dataset = presDataset(pS_array[x_train], pH_array[x_train])
dev_dataset = presDataset(pS_array[x_dev], pH_array[x_dev])
test_dataset = presDataset(pS_array[x_test], pH_array[x_test])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batchSize, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=params.batchSize, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batchSize, shuffle=True)


__all__ = ['sa_data', 'ss_data', 'aa_x1', 'aa_x2', 'aa_edge_index', 'aa_edge_index2',
           'sa_weights', 'ss_weights', 'aa_weights', 'aa_weights2',
           'train_loader', 'dev_loader', 'test_loader', 'x_train', 'x_dev', 'x_test']