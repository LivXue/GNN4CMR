import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter
from torchvision import models

from util import *


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class TxtMLP(nn.Module):
    def __init__(self, code_len=300, txt_bow_len=1386, num_class=24):
        super(TxtMLP, self).__init__()
        self.fc1 = nn.Linear(txt_bow_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.classifier = nn.Linear(code_len, num_class)

    def forward(self, x):
        feat = F.leaky_relu(self.fc1(x), 0.2)
        feat = F.leaky_relu(self.fc2(feat), 0.2)
        predict = self.classifier(feat)
        return feat, predict


class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class ModalClassifier(nn.Module):
    """Network to discriminate modalities"""

    def __init__(self, input_dim=40):
        super(ModalClassifier, self).__init__()
        self.denseL1 = nn.Linear(input_dim, input_dim // 4)
        self.denseL2 = nn.Linear(input_dim // 4, input_dim // 16)
        self.denseL3 = nn.Linear(input_dim // 16, 2)

    def forward(self, x):
        x = ReverseLayerF.apply(x, 1.0)
        out = self.denseL1(x)
        out = self.denseL2(out)
        out = self.denseL3(out)
        return out


class ImgDec(nn.Module):
    """Network to decode image representations"""

    def __init__(self, input_dim=1024, output_dim=4096, hidden_dim=2048):
        super(ImgDec, self).__init__()
        self.denseL1 = nn.Linear(input_dim, hidden_dim)
        self.denseL2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        out = F.relu(self.denseL2(out))
        return out


class TextDec(nn.Module):
    """Network to decode image representations"""

    def __init__(self, input_dim=1024, output_dim=300, hidden_dim=512):
        super(TextDec, self).__init__()
        self.denseL1 = nn.Linear(input_dim, hidden_dim)
        self.denseL2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = F.leaky_relu(self.denseL1(x), 0.2)
        out = F.leaky_relu(self.denseL2(out), 0.2)
        return out


class P_GNN(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024, num_classes=10, in_channel=300, t=0,
                 adj_file=None, inp=None, GNN='GAT', n_layers=4):
        super(P_GNN, self).__init__()
        self.img_net = ImgNN(img_input_dim, minus_one_dim)
        self.text_net = TextNN(text_input_dim, minus_one_dim)
        self.img2text_net = TextDec(minus_one_dim, text_input_dim)
        self.text2img_net = ImgDec(minus_one_dim, img_input_dim)
        self.img_md_net = ModalClassifier(img_input_dim)
        self.text_md_net = ModalClassifier(text_input_dim)
        self.num_classes = num_classes
        if GNN == 'GAT':
            self.gnn = GraphAttentionLayer
        elif GNN == 'GCN':
            self.gnn = GraphConvolution
        else:
            raise NameError("Invalid GNN name!")
        self.n_layers = n_layers

        self.relu = nn.LeakyReLU(0.2)
        self.lrn = [self.gnn(in_channel, minus_one_dim)]
        for i in range(1, self.n_layers):
            self.lrn.append(self.gnn(minus_one_dim, minus_one_dim))
        for i, layer in enumerate(self.lrn):
            self.add_module('lrn_{}'.format(i), layer)
        self.hypo = nn.Linear(self.n_layers * minus_one_dim, minus_one_dim)

        _adj = torch.FloatTensor(gen_A(num_classes, t, adj_file))
        if GNN == 'GAT':
            self.adj = Parameter(_adj, requires_grad=False)
        else:
            self.adj = Parameter(gen_adj(_adj), requires_grad=False)

        if inp is not None:
            self.inp = Parameter(inp, requires_grad=False)
        else:
            self.inp = Parameter(torch.rand(num_classes, in_channel))
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature_img, feature_text):
        view1_feature = self.img_net(feature_img)
        view2_feature = self.text_net(feature_text)

        layers = []
        x = self.inp
        for i in range(self.n_layers):
            x = self.lrn[i](x, self.adj)
            if self.gnn == GraphConvolution:
                x = self.relu(x)
            layers.append(x)
        x = torch.cat(layers, -1)
        x = self.hypo(x)

        norm_img = torch.norm(view1_feature, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        norm_txt = torch.norm(view2_feature, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        x = x.transpose(0, 1)
        y_img = torch.matmul(view1_feature, x)
        y_text = torch.matmul(view2_feature, x)
        y_img = y_img / norm_img
        y_text = y_text / norm_txt

        view1_feature_view2 = self.img2text_net(view1_feature)
        view2_feature_view1 = self.text2img_net(view2_feature)
        view1_modal_view1 = self.img_md_net(feature_img)
        view2_modal_view1 = self.img_md_net(view2_feature_view1)
        view1_modal_view2 = self.text_md_net(view1_feature_view2)
        view2_modal_view2 = self.text_md_net(feature_text)

        return view1_feature, view2_feature, y_img, y_text, x.transpose(0, 1), \
               view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2


class I_GNN(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024, num_classes=10, in_channel=300,
                 t=0.4, k=3, inp=None, GNN='GAT', n_layers=4):
        super(I_GNN, self).__init__()
        self.img_net = ImgNN(img_input_dim, minus_one_dim)
        self.text_net = TextNN(text_input_dim, minus_one_dim)
        self.img2text_net = TextDec(minus_one_dim, text_input_dim)
        self.text2img_net = ImgDec(minus_one_dim, img_input_dim)
        self.img_md_net = ModalClassifier(img_input_dim)
        self.text_md_net = ModalClassifier(text_input_dim)
        self.num_classes = num_classes
        if GNN == 'GAT':
            self.gnn = GraphAttentionLayer
        elif GNN == 'GCN':
            self.gnn = GraphConvolution
        else:
            raise NameError("Invalid GNN name!")
        self.n_layers = n_layers

        self.relu = nn.LeakyReLU(0.2)
        self.lrn = [self.gnn(in_channel, minus_one_dim)]
        for i in range(1, self.n_layers):
            self.lrn.append(self.gnn(minus_one_dim, minus_one_dim))
        for i, layer in enumerate(self.lrn):
            self.add_module('lrn_{}'.format(i), layer)
        self.hypo = nn.Linear(self.n_layers * minus_one_dim, minus_one_dim)

        if inp is None:
            raise NotImplementedError("Category embeddings are missing!")
        self.inp = Parameter(inp, requires_grad=True)
        normalized_inp = F.normalize(inp, dim=1)
        self.A0 = Parameter(torch.matmul(normalized_inp, normalized_inp.T), requires_grad=False)
        self.A0[self.A0 < t] = 0
        self.t = t

        self.Wk = list()
        self.k = k
        for i in range(k):
            Wk_temp = Parameter(torch.zeros(in_channel))
            torch.nn.init.uniform_(Wk_temp.data, 0.3, 0.6)
            self.Wk.append(Wk_temp)
        for i, Wk_temp in enumerate(self.Wk):
            self.register_parameter('W_{}'.format(i), Wk_temp)

        self.lambdda = 0.5

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature_img, feature_text):
        view1_feature = self.img_net(feature_img)
        view2_feature = self.text_net(feature_text)

        S = torch.zeros_like(self.A0)
        for Wk_temp in self.Wk:
            normalized_imp_mul_Wk = F.normalize(self.inp * Wk_temp[None, :], dim=1)
            S += torch.matmul(normalized_imp_mul_Wk, normalized_imp_mul_Wk.T)
        S /= self.k
        S[S < self.t] = 0
        A = self.lambdda * self.A0 + (1-self.lambdda) * S
        adj = gen_adj(A)

        layers = []
        x = self.inp
        for i in range(self.n_layers):
            x = self.lrn[i](x, adj)
            if self.gnn == GraphConvolution:
                x = self.relu(x)
            layers.append(x)
        x = torch.cat(layers, -1)
        x = self.hypo(x)

        norm_img = torch.norm(view1_feature, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        norm_txt = torch.norm(view2_feature, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        x = x.transpose(0, 1)
        y_img = torch.matmul(view1_feature, x)
        y_text = torch.matmul(view2_feature, x)
        y_img = y_img / norm_img
        y_text = y_text / norm_txt

        view1_feature_view2 = self.img2text_net(view1_feature)
        view2_feature_view1 = self.text2img_net(view2_feature)
        view1_modal_view1 = self.img_md_net(feature_img)
        view2_modal_view1 = self.img_md_net(view2_feature_view1)
        view1_modal_view2 = self.text_md_net(view1_feature_view2)
        view2_modal_view2 = self.text_md_net(feature_text)

        return view1_feature, view2_feature, y_img, y_text, x.transpose(0, 1), \
               view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2