import torch.cuda
from torch_snippets import *
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Определим функцию которая будет отвечать за трансформацию и аугментацию (пока нормализацию) датасета
transforms_imgs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
# Это класс датасета, который мы будем использовать для того, чтобы решить задачу сегментации
class SegData(Dataset):
    def __init__(self, split):
        self.items = stems(f'.\data\images_{split}')
        self.split = split

    def __len__(self):
        return len(self.items)

    def __getitem__(self, ix):
        image = read(f'./data/images_{self.split}/{self.items[ix]}.jpg', 1)
        image = cv2.resize(image, (224, 224))
        mask = read(f'./data/annotations_{self.split}/{self.items[ix]}.tif')
        mask = cv2.resize(mask, (224, 224))
        return image, mask

    def choose(self):
        return self[randint(len(self))]

    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        ims = torch.cat([transforms_imgs(im.copy()/255.)[None] for im in ims]).float().to(device)
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(device)
        return ims, ce_masks

class SegDataTest(Dataset):
    def __init__(self, split):
        self.items = stems(f'./data/images_{split}')
        self.split = split

    def __len__(self):
        return len(self.items)

    def __getitem__(self, ix):
        image = read(f'./data/images_for_TEST/test_image.jpg',1)
        image = cv2.resize(image, (224, 224))
        mask = read(f'./data/annotations_for_TEST/annotation.tif')
        mask = cv2.resize(mask, (224, 224))
        return image, mask

    def choose(self):
        return self[randint(len(self))]

    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        ims = torch.cat([transforms_imgs(im.copy()/255.)[None] for im in ims]).float().to(device)
        return ims
