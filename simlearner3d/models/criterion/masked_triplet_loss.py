import torch
import torch.nn.functional as F

class MaskedTripletLoss:
    def __init__(self,margin=0.3):
        self.margin=margin
    def compute_loss(self,anchor, positive, negative,occlusion_masq):
        return torch.max(F.cosine_similarity(anchor,negative)
                                   - F.cosine_similarity(anchor,positive)
                                   +self.margin,0).mul(torch.logical_not(occlusion_masq))+torch.max(
                                       F.cosine_similarity(anchor,negative)
                                        + F.cosine_similarity(anchor,positive)
                                        ,0).mul(occlusion_masq)
    def __call__(self,anchor,positive,negative,occlusion_masq) -> torch.Tensor:
        return self.compute_loss(anchor,
                                 positive,
                                 negative,
                                 occlusion_masq
                                    )
        