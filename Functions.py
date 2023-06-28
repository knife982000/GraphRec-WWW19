import torch

def masked_softmax(vec, mask, dim=1):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec-max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros=(masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/masked_sums


def masked_independent_shuffle(x, mask):
    #creates a shuffle for each part
    shuffle = torch.vstack([torch.torch.randperm(mask.shape[1]).to(mask.device) for _ in range(mask.shape[0])])
    #Generetes the index
    idxs = [torch.arange(mask.shape[0]).unsqueeze(1).repeat(1, mask.shape[1]).to(mask.device), shuffle]
    x = x[idxs]
    mask = mask[idxs]
    #sort according to the mask
    shuffle = torch.argsort(mask, descending=True, stable=True, dim=1)
    idxs[1] = shuffle
    x = x[idxs]
    return x


def masked_shuffle(x, mask):
    #creates a shuffle for each part
    shuffle = torch.torch.randperm(mask.shape[1]).to(mask.device)
    #Generetes the index
    idxs = [torch.arange(mask.shape[0]).unsqueeze(1).repeat(1, mask.shape[1]).to(mask.device), shuffle]
    x = x[idxs]
    mask = mask[idxs]
    #sort according to the mask
    shuffle = torch.argsort(mask, descending=True, stable=True, dim=1)
    idxs[1] = shuffle
    x = x[idxs]
    return x