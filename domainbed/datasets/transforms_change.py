from torchvision import transforms as T


basic = T.Compose(
    [
        T.Resize((224, 224)),
        # T.RandomCrop(224, padding=4),
        # T.CenterCrop(224),
        # T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        # T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        # T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        # T.RandomGrayscale(p=0.1),
        # T.autoaugment.AutoAugment(),
        T.ToTensor(),
        # T.RandomHorizontalFlip(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
aug = T.Compose(
    [
        T.Resize((224, 224)),
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        # T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        # T.RandomGrayscale(p=0.1),
        # T.autoaugment.AutoAugment(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
