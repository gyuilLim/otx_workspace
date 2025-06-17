from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ImageFolderDataModule:
    def __init__(self, root_dir: str, batch_size: int = 64):
        """
        Args:
            root_dir (str): 데이터셋의 루트 디렉토리. 예: './data/CUB-200'
                            하위에 'train', 'val', 'test' 폴더가 있어야 함.
            batch_size (int): 데이터 로더의 배치 크기
        """
        self.root_dir = root_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0],
                std=[58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0],
            ),
        ])

    def setup(self):
        self.train_dataset = datasets.ImageFolder(root=f'{self.root_dir}/train', transform=self.transform)
        self.val_dataset   = datasets.ImageFolder(root=f'{self.root_dir}/val', transform=self.transform)
        self.test_dataset  = datasets.ImageFolder(root=f'{self.root_dir}/test', transform=self.transform)
        self.class_num = len(self.train_dataset.class_to_idx)

    def get_loaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader