

def farthest_point_sampling_by_pointops(vertices,len_vertices:int, n_sample: int = 2000,device="cuda"):
    """ 基于pointops2最远点采样，返回采样后的索引，要求输入为torch.tensor """
    from pointops2.pointops2 import furthestsampling
    import torch
    # 采样
    offset = torch.tensor([0, len_vertices], dtype=torch.int32, device=device)
    new_offset = torch.tensor([0, n_sample], dtype=torch.int32, device=device)
    idx = furthestsampling(vertices.contiguous(), offset, new_offset)
    return idx
