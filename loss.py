import torch
import torch.nn as nn
import torch.nn.functional as F

criterion_md = nn.CrossEntropyLoss()


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim


def cla_loss(view1_predict, view2_predict, labels_1, labels_2):
    cla_loss1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean()
    cla_loss2 = ((view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()

    return cla_loss1 + cla_loss2


def mdl_loss(view1_feature, view2_feature, labels_1, labels_2):
    cos = lambda x, y: x.mm(y.t()) / (
        (x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    Sim22 = calc_label_sim(labels_2, labels_2).float()
    term11 = ((1 + torch.exp(theta11)).log() - Sim11 * theta11).mean()
    term12 = ((1 + torch.exp(theta12)).log() - Sim12 * theta12).mean()
    term22 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    mdl_loss = term11 + term12 + term22

    return mdl_loss


def gan_loss(view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2):
    bs = view1_modal_view1.size()[0]
    img_md = torch.ones(bs, dtype=torch.long).cuda()
    txt_md = torch.zeros(bs, dtype=torch.long).cuda()
    return criterion_md(view1_modal_view1, img_md) + criterion_md(view2_modal_view1, txt_md) + \
           criterion_md(view1_modal_view2, img_md) + criterion_md(view2_modal_view2, txt_md)


def soft_con_loss(view1_feature, view2_feature, labels, t=0.21, gamma=0.13):
    view1_feature = F.normalize(view1_feature, dim=1)
    view2_feature = F.normalize(view2_feature, dim=1)
    # cosine similarity: NxN
    sim_view12 = torch.matmul(view1_feature, view2_feature.T) / t
    sim_view11 = torch.matmul(view1_feature, view1_feature.T) / t
    sim_view22 = torch.matmul(view2_feature, view2_feature.T) / t
    #label_L1 = labels.sum(1)
    #label_sim = torch.matmul(labels, labels.T) / (label_L1[None, :] + label_L1[:, None] - torch.matmul(labels, labels.T))
    label_sim = torch.matmul(labels, labels.T).clamp(max=1.0)
    #label_sim = label_sim ** 0.5
    pro_inter = label_sim / label_sim.sum(1, keepdim=True).clamp(min=1e-6)
    label_sim_intra = (label_sim - torch.eye(label_sim.shape[0]).cuda()).clamp(min=0)
    pro_intra = label_sim_intra / label_sim_intra.sum(1, keepdim=True).clamp(min=1e-6)

    # logits: NxN
    logits_view12 = sim_view12 - torch.log(torch.exp(1.06 * sim_view12).sum(1, keepdim=True))
    logits_view21 = sim_view12.T - torch.log(torch.exp(1.06 * sim_view12.T).sum(1, keepdim=True))
    logits_view11 = sim_view11 - torch.log(torch.exp(1.06 * sim_view11).sum(1, keepdim=True))
    logits_view22 = sim_view22 - torch.log(torch.exp(1.06 * sim_view22).sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos_view12 = (pro_inter * logits_view12).sum(1)
    mean_log_prob_pos_view21 = (pro_inter * logits_view21).sum(1)
    mean_log_prob_pos_view11 = (pro_intra * logits_view11).sum(1)
    mean_log_prob_pos_view22 = (pro_intra * logits_view22).sum(1)

    # supervised cross-modal contrastive loss
    loss = - mean_log_prob_pos_view12.mean() - mean_log_prob_pos_view21.mean() \
           - gamma * (mean_log_prob_pos_view11.mean() + mean_log_prob_pos_view22.mean())

    return loss
