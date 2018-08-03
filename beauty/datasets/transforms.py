import torchvision


def create_transform(transforms):
    transform = torchvision.transforms.Compose([
        transform.transform(**vars(transform.config))
        for transform in transforms
    ])
    return transform
