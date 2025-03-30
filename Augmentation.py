import torch
import numpy as np
import random
import cv2

class Augmentation:
    """
    A custom data augmentation class for 2D medical image segmentation tasks.
    Applies common transformations like flip, rotation, noise, blur, gamma correction,
    and contrast adjustment to both image and label (segmentation mask).
    """

    def __init__(self,
                 p_flip=0.5,        # Probability of applying horizontal flip
                 p_rotate=0.5,      # Probability of applying random rotation
                 p_noise=0.5,       # Probability of adding Gaussian noise
                 p_blur=0.3,        # Probability of applying Gaussian blur
                 p_gamma=0.5,       # Probability of gamma correction
                 p_contrast=0.5,    # Probability of contrast adjustment
                 rotate_angle_range=30,        # Max angle for rotation
                 noise_std=0.05,               # Standard deviation of Gaussian noise
                 gamma_range=(0.7, 1.5),       # Range for gamma values
                 contrast_range=(0.7, 1.3)):   # Range for contrast scaling
        # Set probabilities and parameters
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_noise = p_noise
        self.p_blur = p_blur
        self.p_gamma = p_gamma
        self.p_contrast = p_contrast

        self.rotate_angle_range = rotate_angle_range
        self.noise_std = noise_std
        self.gamma_range = gamma_range
        self.contrast_range = contrast_range

    def __call__(self, sample):
        """
        Apply augmentations to a sample (dictionary containing 'image' and 'label').
        Args:
            sample: dict with keys 'image' and 'label', both are torch tensors with shape [1, H, W]
        Returns:
            Augmented sample as a dict with the same keys and shapes.
        """
        image, label = sample['image'], sample['label']
        
        # Convert tensors to NumPy arrays and remove channel dimension
        image_np = image.squeeze(0).numpy()
        label_np = label.squeeze(0).numpy()

        # 1. Random Horizontal Flip
        if random.random() < self.p_flip:
            image_np = np.fliplr(image_np).copy()
            label_np = np.fliplr(label_np).copy()

        # 2. Random Rotation
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.rotate_angle_range, self.rotate_angle_range)
            center = (image_np.shape[1] / 2, image_np.shape[0] / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
            # Apply affine transform to image and label
            image_np = cv2.warpAffine(image_np, M, (image_np.shape[1], image_np.shape[0]), flags=cv2.INTER_LINEAR)
            label_np = cv2.warpAffine(label_np, M, (label_np.shape[1], label_np.shape[0]), flags=cv2.INTER_NEAREST)

        # 3. Add Gaussian Noise
        if random.random() < self.p_noise:
            noise = np.random.normal(0, self.noise_std, image_np.shape)
            image_np += noise

        # 4. Apply Gaussian Blur
        if random.random() < self.p_blur:
            k = random.choice([3, 5])  # Kernel size must be odd
            image_np = cv2.GaussianBlur(image_np, (k, k), sigmaX=0)

        # 5. Gamma Correction
        if random.random() < self.p_gamma:
            gamma = random.uniform(*self.gamma_range)
            image_np = np.clip(image_np, 1e-6, 1.0)  # Avoid invalid power ops
            image_np = np.power(image_np, gamma)

        # 6. Contrast Adjustment
        if random.random() < self.p_contrast:
            factor = random.uniform(*self.contrast_range)
            mean = image_np.mean()
            image_np = (image_np - mean) * factor + mean

        # Final cleanup:
        # Replace any NaN/inf with safe values and clip to [0,1]
        image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
        image_np = np.clip(image_np, 0, 1)

        # Convert back to torch tensors with shape [1, H, W]
        image_aug = torch.tensor(image_np[np.newaxis, ...], dtype=torch.float32)
        label_aug = torch.tensor(label_np[np.newaxis, ...], dtype=torch.long)

        return {'image': image_aug, 'label': label_aug}
