import random

from torchvision import transforms
import torchvision.transforms.functional as TF


class Augmentation:
    def __init__(self, p):
        self.p = p

    def augmentation_function(self, input_tensor):
        raise NotImplementedError("Augmentation function is not defined")

    def __call__(self, image, gt):
        if random.random() < self.p:
            return self.augmentation_function(image, gt)

        return image, gt


class AugmentationsCompose:
    def __init__(self):
        self.augmentations = []

    def append(self, augmentation):
        self.augmentations.append(augmentation)

    def __call__(self, image, gt):
        for augmentation in self.augmentations:
            image, gt = augmentation(image, gt)

        return image, gt


class GaussianBlur(Augmentation):
    def __init__(self, p, kernel_size, sigma):
        super().__init__(p)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def augmentation_function(self, image, gt):
        kernel_size = random.randrange(*self.kernel_size, 2)
        sigma = random.uniform(*self.sigma)

        return TF.gaussian_blur(image, kernel_size, sigma), gt


class RandomHorizontalFlip(Augmentation):
    def __init__(self, p):
        super().__init__(p)

    def augmentation_function(self, image, gt):
        return TF.hflip(image), TF.hflip(gt)


class RandomVerticalFlip(Augmentation):
    def __init__(self, p):
        super().__init__(p)

    def augmentation_function(self, image, gt):
        return TF.vflip(image), TF.vflip(gt)


class AdjustGamma(Augmentation):
    def __init__(self, p, gamma):
        super().__init__(p)
        self.gamma = gamma

    def augmentation_function(self, image, gt):
        gamma = random.uniform(*self.gamma)
        return TF.adjust_gamma(image, gamma=gamma), gt


class AdjustHue(Augmentation):
    def __init__(self, p, hue):
        super().__init__(p)
        self.hue = hue

    def augmentation_function(self, image, gt):
        hue = random.uniform(*self.hue)
        return TF.adjust_hue(image, hue_factor=hue), gt


class AdjustContrast(Augmentation):
    def __init__(self, p, contrast):
        super().__init__(p)
        self.contrast = contrast

    def augmentation_function(self, image, gt):
        contrast = random.uniform(*self.contrast)
        return TF.adjust_contrast(image, contrast_factor=contrast), gt


def get_augmentations(args):
    augmentations = AugmentationsCompose()

    if args.aug_gaussian_blur:
        augmentations.append(GaussianBlur(p=0.2, kernel_size=(5, 9), sigma=(0.1, 5)))

    if args.aug_horizontal_flip:
        augmentations.append(RandomHorizontalFlip(p=0.2))

    if args.aug_vertical_flip:
        augmentations.append(RandomVerticalFlip(p=0.2))

    if args.aug_adjust_gamma:
        augmentations.append(AdjustGamma(p=0.2, gamma=(0.4, 1.6)))

    if args.aug_adjust_hue:
        augmentations.append(AdjustHue(p=0.2, hue=(-0.5, 0.5)))

    if args.aug_adjust_contrast:
        augmentations.append(AdjustContrast(p=0.2, contrast=(0.4, 1.6)))

    return augmentations
