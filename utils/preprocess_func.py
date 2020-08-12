import albumentations as albu
def get_training_augmentation():
    train_transform = [

        albu.IAAFlipud(always_apply=False, p=0.5),
        albu.IAAFliplr(always_apply=False, p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, interpolation=2, border_mode=4, always_apply=False, p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, interpolation=2, border_mode=4, always_apply=False, p=0.5),
        albu.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=0, interpolation=2, border_mode=4, always_apply=False, p=0.5),
        albu.RandomBrightness(limit=(0.8,1.2), p=0.5),
        albu.ElasticTransform(alpha=720,sigma=24,alpha_affine=24,approximate=True,p=0.5),
        albu.IAAAffine(scale=1.0, translate_percent=None, translate_px=None, rotate=0.0, shear=0.2, order=1, cval=0, mode='reflect', always_apply=False, p=1),
        albu.PadIfNeeded(256, 256)
    ]

    '''train_transform = [
        albu.IAAAdditiveGaussianNoise(p=0.03),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, interpolation=2, border_mode=4,
                              always_apply=False, p=0.6),
        albu.RandomBrightnessContrast(brightness_limit=0.02, contrast_limit=0.02, always_apply=False, p=0.7),
        albu.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, always_apply=False, p=0.1),
        albu.ElasticTransform(sigma = 10, alpha_affine=10,approximate=True,p=0.5)
    ]'''
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(256, 256)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)