import torch
import torch.nn as nn
import torch.nn.functional as F

# The implemented ArcFace loss function replaces the classification head of the original model, taking in the cls token.
class ArcFace(nn.Module):
    """
    Implementation of the ArcFace loss function: https://arxiv.org/pdf/1801.07698.pdf
    based on: https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch/blob/master/loss_functions.py
    """
    def __init__(self, in_features, out_features, eps=1e-7, s=64., m=0.5):
        super().__init__()
        print('using ArcFace')
        self.in_fearues = in_features
        self.out_features = out_features
        self.eps = eps
        self.s = s
        self.m = m
        self.fc = nn.Linear(in_features, out_features, bias=False)
        

    def forward(self, x, labels):
        """ 
        the input x is expected to be of dimensions (batch_size, in_features) 
        """
        for weight in self.fc.parameters():
            weight = F.normalize(weight, p=2, dim=1)
        with torch.no_grad():
            output = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        
        cosines = self.fc(x)

        # clamp for numerical stability
        numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(cosines.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
       
        # all the cosines for the rest of the classes
        rest = torch.cat([torch.cat((cosines[i, :y], cosines[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * rest), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L), output
    
class TripletLoss(nn.Module):
    """
    Triplet loss from: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/loss.py
    """

    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x, label):
        return triplet_loss(x, label, margin=self.margin)
    
    
def triplet_loss(x, label, margin):
    # x is D x N
    dim = x.size(0) # Dimension of the models output
    nq = torch.sum(label.data==-1).item() # number of tuples / number of batches
    S = x.size(1) // nq # number of images per tuple including query: 1+1+n

    xa = x[:, label.data==-1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    xp = x[:, label.data==1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    xn = x[:, label.data==0]

    dist_pos = torch.sum(torch.pow(xa - xp, 2), dim=0)
    dist_neg = torch.sum(torch.pow(xa - xn, 2), dim=0)

    return torch.sum(torch.clamp(dist_pos - dist_neg + margin, min=0))

class TripletLossTwoInputs(nn.Module):
    """
    Modified version of the triplet loss calculating the contribution of both output token to the global loss.
    Additionally a cosine loss meant to enforce some dissimilarity between the outputs is implemented (only for the anchors at this point)
    """
    def __init__(self, margin=0.1, alpha=0.5):
        super(TripletLossTwoInputs, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, x1, x2, label):
         # x is D x N
        dim = x1.size(0) # Dimension of the models output
        nq = torch.sum(label.data==-1).item() # number of tuples / number of batches
        S = x1.size(1) // nq # number of images per tuple including query: 1+1+n

        x1a = x1[:, label.data==-1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
        x1p = x1[:, label.data==1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
        x1n = x1[:, label.data==0]
        
        x2a = x2[:, label.data==-1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
        x2p = x2[:, label.data==1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
        x2n = x2[:, label.data==0]

        dist_pos1 = torch.sum(torch.pow(x1a - x1p, 2), dim=0)
        dist_neg1 = torch.sum(torch.pow(x1a - x1n, 2), dim=0)

        dist_pos2 = torch.sum(torch.pow(x1a - x1p, 2), dim=0)
        dist_neg2 = torch.sum(torch.pow(x1a - x1n, 2), dim=0)
        
        loss_first_token = torch.sum(torch.clamp(dist_pos1 - dist_neg1 + self.margin, min=0))
        loss_second_token = torch.sum(torch.clamp(dist_pos2 - dist_neg2 + self.margin, min=0))
        
        # only for anchors for now
        cosine_loss = torch.clamp(self.cos(x1a[:,0], x2a[:,0]) - 0.9, min=0)

        return (loss_first_token * self.alpha + loss_second_token * (1 - self.alpha)) + cosine_loss


    
    
    

