import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50
from nystrom_attention import NystromAttention
import torch.nn.functional as F


class CustomAttention(nn.Module):
    def __init__(
        self,
        weight_params_dim,
        **kwargs
    ):
        super(CustomAttention, self).__init__(**kwargs)

        self.wq = nn.Linear(512, weight_params_dim)
        self.wk = nn.Linear(512, weight_params_dim)


    def forward(self, inputs):
        wsi_bag = inputs
        attention_weights = self.compute_attention_scores(wsi_bag)
        return attention_weights

    def compute_attention_scores(self, instance):
        q = self.wq(instance)
        k = self.wk(instance)

        dk = torch.tensor(k.shape[-1], dtype=torch.float32)

        matmul_qk = torch.matmul(q, k.transpose(-2, -1))

        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        return scaled_attention_logits

class SparseMatrix(nn.Module):
    def __init__(self, k, **kwargs):
        super(SparseMatrix, self).__init__(**kwargs)
        self.k = k


    def get_sparse_matrix(self, neighbor_indices, values, k):
        with torch.no_grad():
            device = neighbor_indices.device

            neighbor_indices = neighbor_indices[:, :k]
            values = values[:, :k]

            tile_ids, sp_cols, counts = torch.unique(neighbor_indices.flatten(), return_counts=True, return_inverse=True)

            sp_rows = torch.sort(torch.cat(k * [torch.arange(neighbor_indices.size(0))], dim=0), dim=0)[0].to(device)

            indices = torch.stack([sp_rows, sp_cols], dim=0)

            values = values.flatten().to(device)

            sparse_matrix =  torch.sparse_coo_tensor(indices, values, dtype=torch.double).to(device)
            sparse_matrix = sparse_matrix.coalesce()

        return sparse_matrix

    def forward(self, inputs):
        indices = inputs[0]
        values = inputs[1]

        affinity_matrix = self.get_sparse_matrix(indices, values, self.k)
        return affinity_matrix


class NeighborAggregator(nn.Module):
    def __init__(self, output_dim, k, **kwargs):
        super(NeighborAggregator, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.sparse_layer = SparseMatrix(k=k)

    def sparse_dense_mul(self, s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    def forward(self, inputs):
        input_tensor = inputs[0]
        indices = inputs[1]
        values = inputs[2]

        affinity_matrix = self.sparse_layer([indices, values])

        sparse_data_input = self.sparse_dense_mul(affinity_matrix, input_tensor.double())

        reduced_sum = (torch.sparse.sum(sparse_data_input, dim=0))

        alpha = F.softmax (reduced_sum.values(), dim=0)

        return alpha, reduced_sum.values()



class feature_extractor(nn.Module):
    def __init__(self, weight_decay=None):
        super(feature_extractor, self).__init__()
        self.custom_att = CustomAttention(weight_params_dim=256)
        self.wv = nn.Linear(512, 512)
        self.nyst_att = NystromAttention(dim = 512,dim_head = 64,heads = 8,num_landmarks = 512)
        self.weight_decay=weight_decay
        self.attcls = Attention_Gated(weight_decay=weight_decay)

        self.dense = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )

    def forward(self, inputs):
        bag_tensor = inputs
        # indices = inputs[1]
        # values = inputs[2]

        dense = self.dense(bag_tensor)
        # encoder_output = torch.squeeze(self.nyst_att(torch.unsqueeze(dense, dim=0)))
        # encoder_output = encoder_output.view(-1, 512)
        # encoder_output = dense + encoder_output
        #
        # attention_matrix = self.custom_att(encoder_output)
        # norm_alpha, alpha = self.neigh([attention_matrix, indices, values])
        #
        # value = self.wv(dense)
        #
        # local_attn_output =  norm_alpha.unsqueeze(1) * value
        #
        # local_attn_output = local_attn_output + encoder_output

        if self.weight_decay is not None:
            k_alpha, reg = self.attcls(dense.float())
        else:
            k_alpha = self.attcls(dense.float())

        attn_output = torch.mm(k_alpha, dense)

        return attn_output


class Encoder(nn.Module):
    def __init__(self, n_classes, k, weight_decay=None):
        super(Encoder, self).__init__()
        self.custom_att = CustomAttention(weight_params_dim=256)
        self.wv = nn.Linear(512, 512)
        self.nyst_att = NystromAttention(dim = 512,dim_head = 64,heads = 8,num_landmarks = 512)
        self.k = k
        self.n_classes=n_classes
        self.weight_decay=weight_decay
        self.attcls = Attention_Gated(weight_decay=weight_decay)
        self.neigh = NeighborAggregator(output_dim=1, k=self.k)

        self.classifier = nn.Linear(512, self.n_classes)

        self.dense = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )

    def forward(self, inputs):
        bag_tensor = inputs[0]
        indices = inputs[1]
        values = inputs[2]

        dense = self.dense(bag_tensor)
        # encoder_output = torch.squeeze(self.nyst_att(torch.unsqueeze(dense, dim=0)))
        # encoder_output = encoder_output.view(-1, 512)
        # encoder_output = dense + encoder_output
        #
        # attention_matrix = self.custom_att(encoder_output)
        # norm_alpha, alpha = self.neigh([attention_matrix, indices, values])
        #
        # value = self.wv(dense)
        #
        # local_attn_output =  norm_alpha.unsqueeze(1) * value
        #
        # local_attn_output = local_attn_output + encoder_output

        if self.weight_decay is not None:
            k_alpha, reg = self.attcls(dense.float())
        else:
            k_alpha = self.attcls(dense.float())

        attn_output = torch.mm(k_alpha, dense)

        logits = self.classifier(attn_output)

        if self.weight_decay is not None:
            regularization_loss = self.weight_decay * torch.norm(self.classifier.weight, p=2)
            Y_prob = torch.sigmoid(logits)
            return Y_prob, regularization_loss + reg

        Y_prob = torch.softmax(logits, dim=1)

        return Y_prob

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1, weight_decay=None):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.weight_decay = weight_decay

        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, K)
    def forward(self, x, isNorm=True):

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N



