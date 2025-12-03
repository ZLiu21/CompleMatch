import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def cross_entropy():
    loss = nn.CrossEntropyLoss()
    return loss


def reconstruction_loss():
    loss = nn.MSELoss()
    return loss


def sup_contrastive_loss(embd_batch, labels, device,
                         temperature=0.07, base_temperature=0.07):
    anchor_dot_contrast = torch.div(
        torch.matmul(embd_batch, embd_batch.T),
        temperature)

    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.scatter(
        torch.ones_like(logits.detach()),
        1,
        torch.arange(embd_batch.shape[0]).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    num_anchor = 0
    for s in mask.sum(1):
        if s != 0:
            num_anchor = num_anchor + 1
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.sum(0) / (num_anchor + 1e-12)

    return loss


def reg_co_training_loss(time_pre, freq_pre, threshold=0.8, k=15, phi=1.0): 
    prob_time = F.softmax(time_pre, dim=1)
    prob_freq = F.softmax(freq_pre, dim=1)

    max_time = prob_time.max(dim=1)[0]
    max_freq = prob_freq.max(dim=1)[0]

    conf_time = torch.sigmoid(k * (max_time - threshold))
    conf_freq = torch.sigmoid(k * (max_freq - threshold))

    P_T_and_F = (conf_time * conf_freq).mean()

    P_not_T_and_not_F = ((1 - conf_time) * (1 - conf_freq)).mean()

    P_xor = (conf_time * (1 - conf_freq) + conf_freq * (1 - conf_time)).mean()

    approx_min = -torch.log(torch.exp(-P_T_and_F) + torch.exp(-P_not_T_and_not_F))

    L_con = phi * approx_min - P_xor

    return L_con


class SimCLRContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(SimCLRContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)

        similarity_scores = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1) / self.temperature

        labels = torch.arange(len(z1), device=z1.device)
        contrastive_loss = F.cross_entropy(similarity_scores, labels)
        
        return contrastive_loss

