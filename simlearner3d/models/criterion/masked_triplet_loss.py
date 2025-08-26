import torch
import torch.nn.functional as F

class MaskedTripletLoss:
    def __init__(self,margin=0.3):
        self.margin=margin
    def compute_loss(self,anchor, positive, negative,occlusion_masq):
        return torch.clamp_min(F.cosine_similarity(anchor,negative)
                                   - F.cosine_similarity(anchor,positive)
                                   +self.margin,0).mul(torch.logical_not(occlusion_masq))+(torch.clamp_min(
                                       F.cosine_similarity(anchor,negative),
                                       0) + torch.clamp_min(
                                           F.cosine_similarity(anchor,positive)
                                        ,0)).mul(occlusion_masq)
    def __call__(self,anchor,positive,negative,occlusion_masq) -> torch.Tensor:
        return self.compute_loss(anchor,
                                 positive,
                                 negative,
                                 occlusion_masq
                                    )
    
class NPairLoss:
    def __init__(self, margin =0.3):
        self.margin= margin 
    def __call__(self, matching, non_matching, occlusion_masq) -> torch.Tensor:
        return torch.clamp_min(non_matching - matching + self.margin,0)
        
class NMaskedPairLoss:
    def __init__(self, margin =0.3):
        self.margin= margin 
    def __call__(self, matching, non_matching, masq) -> torch.Tensor:
        return torch.clamp_min(non_matching - matching + self.margin,0).mul(masq)

class SimpleTripletLoss:
    def __init__(self,margin=0.3):
        self.margin=margin
    def compute_loss(self,anchor, positive, negative):
        return torch.clamp_min(F.cosine_similarity(anchor,negative)
                                   - F.cosine_similarity(anchor,positive)
                                   +self.margin,0)
    def __call__(self,anchor,positive,negative,) -> torch.Tensor:
        return self.compute_loss(anchor,
                                 positive,
                                 negative
                                    )