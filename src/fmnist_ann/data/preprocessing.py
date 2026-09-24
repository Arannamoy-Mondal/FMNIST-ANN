from torchvision import transforms


def get_train_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)),
        ]
    )


def get_test_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)),
        ]
    )