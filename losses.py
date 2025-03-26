import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.squeeze(1), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        union = torch.sum(probs + targets_one_hot, dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1. - dice.mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target.squeeze(1), weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class BCE_Dice_Loss(nn.Module):
    def __init__(self, weight_dice=0.5):
        super(BCE_Dice_Loss, self).__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        loss_dice = self.dice(logits, targets)
        loss_ce = self.ce(logits, targets.squeeze(1))
        return self.weight_dice * loss_dice + (1 - self.weight_dice) * loss_ce

class Focal_Dice_Loss(nn.Module):
    def __init__(self, gamma=2.0, weight_dice=0.5):
        super(Focal_Dice_Loss, self).__init__()
        self.focal = FocalLoss(gamma=gamma)
        self.dice = DiceLoss()
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.weight_dice * loss_dice + (1 - self.weight_dice) * loss_focal

class ComboLoss(nn.Module):
    def __init__(self, first, second, weight=0.5):
        super(ComboLoss, self).__init__()
        self.first = first  
        self.second = second  
        self.weight = weight

    def forward(self, logits, targets):
        
        if logits.shape[2:] != targets.shape[2:]:
            targets = F.interpolate(targets.float(), size=logits.shape[2:], mode='nearest').long()

        loss_dice = self.first(logits, targets)
        loss_other = self.second(logits, targets.squeeze(1))  
        return self.weight * loss_dice + (1 - self.weight) * loss_other

