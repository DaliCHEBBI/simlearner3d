import torch
import torch.nn.functional as F

class MaskedTripletLoss:
    def __init__(self,anchor,positive,negative,margin,occlusion_masq):
        self.anchor=anchor
        self.positive=positive
        self.negative=negative
        self.margin=margin
        self.occlusion_masq=occlusion_masq

    def __call__(self) -> torch.Tensor:
        return torch.max(F.cosine_similarity(self.anchor,self.negative)
                                   - F.cosine_similarity(self.anchor,self.positive)
                                   +self.margin,0).mul(torch.logical_not(self.occlusion_masq))+torch.max(
                                       F.cosine_similarity(self.anchor,self.negative)
                                        + F.cosine_similarity(self.anchor,self.positive)
                                        ,0).mul(self.occlusion_masq)
        