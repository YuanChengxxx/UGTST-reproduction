import torch
import numpy as np
import random
import cv2

class Augmentation:
    def __init__(self,
                 p_flip=0.5,
                 p_rotate=0.5,
                 p_noise=0.5,
                 p_blur=0.3,
                 p_gamma=0.5,
                 p_contrast=0.5,
                 rotate_angle_range=30,
                 noise_std=0.05,
                 gamma_range=(0.7, 1.5),
                 contrast_range=(0.7, 1.3)):

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
        image, label = sample['image'], sample['label']  # Tensors: [1, H, W]
        image_np = image.squeeze(0).numpy()
        label_np = label.squeeze(0).numpy()

        # Random Flip
        if random.random() < self.p_flip:
            image_np = np.fliplr(image_np).copy()
            label_np = np.fliplr(label_np).copy()

        # Random Rotation
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.rotate_angle_range, self.rotate_angle_range)
            M = cv2.getRotationMatrix2D((image_np.shape[1] / 2, image_np.shape[0] / 2), angle, 1.0)
            image_np = cv2.warpAffine(image_np, M, (image_np.shape[1], image_np.shape[0]), flags=cv2.INTER_LINEAR)
            label_np = cv2.warpAffine(label_np, M, (label_np.shape[1], label_np.shape[0]), flags=cv2.INTER_NEAREST)

        # Add Gaussian Noise
        if random.random() < self.p_noise:
            noise = np.random.normal(0, self.noise_std, image_np.shape)
            image_np += noise

        # Apply Gaussian Blur
        if random.random() < self.p_blur:
            k = random.choice([3, 5])
            image_np = cv2.GaussianBlur(image_np, (k, k), 0)

        # Gamma Correction
        if random.random() < self.p_gamma:
            gamma = random.uniform(*self.gamma_range)
            image_np = np.clip(image_np, 1e-6, 1.0)
            image_np = np.power(image_np, gamma)

        # Contrast Adjustment
        if random.random() < self.p_contrast:
            factor = random.uniform(*self.contrast_range)
            mean = image_np.mean()
            image_np = (image_np - mean) * factor + mean

        # Final cleanup to avoid nan/inf values
        image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
        image_np = np.clip(image_np, 0, 1)

        image_aug = torch.tensor(image_np[np.newaxis, ...], dtype=torch.float32)
        label_aug = torch.tensor(label_np[np.newaxis, ...], dtype=torch.long)

        return {'image': image_aug, 'label': label_aug}
