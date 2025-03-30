import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation.
    Encourages overlap between predicted and ground truth masks.
    """

    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # Small constant to avoid division by zero

    def forward(self, logits, targets):
        # Apply softmax to get class probabilities
        probs = F.softmax(logits, dim=1)
        
        # Convert target labels to one-hot encoding
        # targets: (N, 1, H, W) -> (N, C, H, W)
        targets_one_hot = F.one_hot(targets.squeeze(1), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        
        dims = (0, 2, 3)  # Dimensions to reduce over: batch, height, width
        
        # Calculate intersection and union
        intersection = torch.sum(probs * targets_one_hot, dims)
        union = torch.sum(probs + targets_one_hot, dims)
        
        # Compute Dice score
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss = 1 - mean Dice score
        loss = 1. - dice.mean()
        return loss

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Modifies cross-entropy to focus learning on hard examples.
    """

    def __init__(self, gamma=2.0, weight=None, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # Focusing parameter for modulating factor
        self.weight = weight  # Optional class weights
        self.ignore_index = ignore_index  # Index to ignore in loss computation

    def forward(self, input, target):
        # Compute standard cross-entropy loss (per-pixel, not reduced)
        ce_loss = F.cross_entropy(input, target.squeeze(1), weight=self.weight,
                                  ignore_index=self.ignore_index, reduction='none')
        
        # pt is the probability of the true class
        pt = torch.exp(-ce_loss)
        
        # Apply focal loss scaling
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class BCE_Dice_Loss(nn.Module):
    """
    Combined Dice and Cross-Entropy Loss.
    The balance between them is controlled by weight_dice.
    """

    def __init__(self, weight_dice=0.5):
        super(BCE_Dice_Loss, self).__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.weight_dice = weight_dice  # Weight for Dice loss vs CE loss

    def forward(self, logits, targets):
        loss_dice = self.dice(logits, targets)
        loss_ce = self.ce(logits, targets.squeeze(1))  # Remove channel dim for CE
        return self.weight_dice * loss_dice + (1 - self.weight_dice) * loss_ce

class Focal_Dice_Loss(nn.Module):
    """
    Combined Dice and Focal Loss.
    Helps with both overlap and class imbalance.
    """

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
    """
    Generic combined loss using two custom loss functions.
    Supports resizing of targets to match prediction shape if needed.
    """

    def __init__(self, first, second, weight=0.5):
        super(ComboLoss, self).__init__()
        self.first = first    # First loss function (e.g., DiceLoss)
        self.second = second  # Second loss function (e.g., CrossEntropyLoss)
        self.weight = weight  # Weight for combining the two losses

    def forward(self, logits, targets):
        # Resize targets if shape doesn't match prediction (e.g., in multi-scale networks)
        if logits.shape[2:] != targets.shape[2:]:
            targets = F.interpolate(targets.float(), size=logits.shape[2:], mode='nearest').long()

        loss_dice = self.first(logits, targets)  # Dice or other segmentation loss
        loss_other = self.second(logits, targets.squeeze(1))  # Usually CE, remove channel dim
        return self.weight * loss_dice + (1 - self.weight) * loss_other
