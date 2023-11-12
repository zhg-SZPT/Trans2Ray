import torch


def batch_index_select(x,idx):
    if len(x.size()) == 3:
        N, B, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset

        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError

def get_index(idx, h1, h2):#这部分应该是找到对应的细粒度划分的重要patch的标号
    '''
    get index of fine stage corresponding to coarse stage
    '''
    H1 = h1
    H2 = h2
    y = idx%H1
    idx1 = 4*idx - 2*y
    idx2 = idx1 + 1
    idx3 = idx1 + H2
    idx4 = idx3 + 1
    idx_finnal = torch.cat((idx1,idx2,idx3,idx4),dim=1)   # transformer对位置不敏感，位置随意
    return idx_finnal