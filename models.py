import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class GCNConv_SA(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_SA, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.tanh(out)

    def message(self, x_j, edge_weight):
        x_j = self.lin(x_j)

        weighted_xj = x_j * edge_weight.view(-1, 1)

        return weighted_xj


class GCNConv_SS_AA(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_SS_AA, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.tanh(out)

    def message(self, x_j, edge_weight):
        x_j = self.lin(x_j)

        weighted_xj = x_j * edge_weight.view(-1, 1)

        return weighted_xj


class HyPoHGF(torch.nn.Module):
    def __init__(self, all_num, embedding_dim, out_channels, batchSize, drop):

        super(HyPoHGF, self).__init__()
        self.batchSize = batchSize
        self.dropout = drop
        self.All_embedding = torch.nn.Embedding(all_num, embedding_dim)
        self.convSA_TostudyS_1 = GCNConv_SA(embedding_dim, embedding_dim)
        self.convSA_TostudyS_2 = GCNConv_SA(embedding_dim, embedding_dim)

        self.SA_mlp_1 = torch.nn.Linear(embedding_dim, out_channels)
        self.SA_bn_1 = torch.nn.BatchNorm1d(out_channels)
        self.SA_tanh_1 = torch.nn.Tanh()

        self.convSA_TostudyS_1_a = GCNConv_SA(embedding_dim, embedding_dim)

        self.convSA_TostudyS_2_a = GCNConv_SA(embedding_dim, embedding_dim)

        self.SA_mlp_1_a = torch.nn.Linear(embedding_dim, out_channels)
        self.SA_bn_1_a = torch.nn.BatchNorm1d(out_channels)
        self.SA_tanh_1_a = torch.nn.Tanh()

        self.convSS = GCNConv_SS_AA(embedding_dim, out_channels)

        self.convAA_1 = GCNConv_SS_AA(embedding_dim, out_channels)
        self.convAA_2 = GCNConv_SS_AA(embedding_dim, out_channels)

        self.mlp = torch.nn.Linear(out_channels, out_channels)
        self.SI_bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()

        self.weight_es = torch.nn.Parameter(torch.randn(2))
        self.weight_eh = torch.nn.Parameter(torch.randn(3))
        self.SI_ln = torch.nn.LayerNorm(out_channels)
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1))
        self.alpha = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x_SA, edge_index_SA, edge_attr_SA, x_SS, edge_index_SS, edge_attr_SS, x_AA1, x_AA2, edge_index_AA,
                edge_index_AA2, edge_attr_AA, edge_attr_AA2, prescription):

        x_SA1 = self.All_embedding(x_SA.long())

        x_SA2 = self.convSA_TostudyS_1(x_SA1.float(), edge_index_SA, edge_attr_SA)

        x_SA6 = self.convSA_TostudyS_2(x_SA2, edge_index_SA, edge_attr_SA)

        x_SA9 = (x_SA1 + x_SA2 + x_SA6) / 3

        x_SA9 = self.SA_mlp_1(x_SA9)
        x_SA9 = self.SA_bn_1(x_SA9)
        x_SA9 = self.SA_tanh_1(x_SA9)

        x_SA11 = self.All_embedding(x_SA.long())

        x_SA22 = self.convSA_TostudyS_1_a(x_SA11.float(), edge_index_SA, edge_attr_SA)

        x_SA66 = self.convSA_TostudyS_2_a(x_SA22, edge_index_SA, edge_attr_SA)

        x_SA99 = (x_SA11 + x_SA22 + x_SA66) / 3
        x_SA99 = self.SA_mlp_1_a(x_SA99)

        x_SA99 = self.SA_bn_1_a(x_SA99)
        x_SA99 = self.SA_tanh_1_a(x_SA99)

        x_ss0 = self.All_embedding(x_SS.long())
        x_ss1 = self.convSS(x_ss0.float(), edge_index_SS, edge_attr_SS)

        x_aa0_1 = self.All_embedding(x_AA1.long())
        x_aa0_2 = self.All_embedding(x_AA2.long())
        x_aa0_2 = torch.cat([x_aa0_1, x_aa0_2], dim=0)

        x_aa1_1 = self.convAA_1(x_aa0_1.float(), edge_index_AA, edge_attr_AA)
        x_aa1_2 = self.convAA_2(x_aa0_2.float(), edge_index_AA2, edge_attr_AA2)
        x_aa1_2 = x_aa1_2[:414]

        weight_es_softmax = torch.nn.functional.softmax(self.weight_es, dim=0)
        weight_eh_softmax = torch.nn.functional.softmax(self.weight_eh, dim=0)

        es_weighted = weight_es_softmax[0] * x_SA9[:466] + weight_es_softmax[1] * x_ss1
        eh_weighted = weight_eh_softmax[0] * x_SA99[466:] + weight_eh_softmax[1] * x_aa1_1 + weight_eh_softmax[
            2] * x_aa1_2

        batch_size = prescription.size(0)
        feature_dim = es_weighted.size(1)
        device = prescription.device

        es_weighted = es_weighted.to(device)

        mean_features = torch.zeros(batch_size, feature_dim, device=device)
        attn_features = torch.zeros(batch_size, feature_dim, device=device)

        for i in range(batch_size):

            symptom_indices = torch.where(prescription[i] > 0)[0].numpy()
            active_nodes = [int(idx) for idx in symptom_indices]
            num_symptoms = len(active_nodes)

            if num_symptoms == 0:
                continue

            symptom_embeddings = es_weighted[active_nodes]

            mean_pooled = torch.mean(symptom_embeddings, dim=0)
            mean_features[i] = mean_pooled

            attention_scores = self.attention(symptom_embeddings)
            attention_weights = F.softmax(attention_scores, dim=0)
            attn_pooled = torch.sum(symptom_embeddings * attention_weights, dim=0)
            attn_features[i] = attn_pooled

        e_synd_mean = self.mlp(mean_features)
        e_synd_mean = self.SI_bn(e_synd_mean)
        e_synd_mean = self.relu(e_synd_mean)
        pre_mean = torch.mm(e_synd_mean, eh_weighted.t())

        e_synd_attn = self.mlp(attn_features)
        e_synd_attn = self.SI_bn(e_synd_attn)
        e_synd_attn = self.relu(e_synd_attn)
        pre_attn = torch.mm(e_synd_attn, eh_weighted.t())

        batch_size, num_acupoints = pre_mean.shape

        sorted_indices = torch.argsort(pre_mean, dim=1, descending=True)
        ranks = torch.zeros_like(pre_mean, dtype=torch.float32, device=device)

        for i in range(batch_size):
            ranks[i, sorted_indices[i]] = torch.arange(1, num_acupoints + 1, device=device).float()

        alpha = F.softplus(self.alpha)
        weights = torch.exp(-alpha * (ranks - 1))

        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        pre_mean_weighted = pre_mean * weights

        pre = pre_attn + pre_mean_weighted

        return pre