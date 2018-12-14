import torchvision


def create_transform(transforms):
    transform = torchvision.transforms.Compose([
        transform.transform(**vars(transform.config))
        for transform in transforms
    ])
    return transform


class ToColor:

    def __call__(self, image):
        color_image = image.convert('RGB')
        return color_image

