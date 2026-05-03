import random

from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import torch.multiprocessing
import json

# import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np

torch.multiprocessing.set_sharing_strategy('file_system')


class RandomLightingAugmentation:
    """Randomly apply one lighting augmentation per image.
    - Directional lighting (left/right/top/bottom)
    - Over/underexposure
    - Color tint and warmth
    Applied with a given probability; otherwise image is unchanged."""

    def __init__(self, p=0.5, intensity_range=(0.15, 0.4)):
        self.p = p
        self.intensity_range = intensity_range

    def _apply_one(self, img_np, aug, intensity):
        if aug in ('left', 'right', 'top', 'bottom'):
            h, w = img_np.shape[:2]
            if aug == 'left':
                grad = np.linspace(1, 0, w)[None, :].repeat(h, axis=0)
            elif aug == 'right':
                grad = np.linspace(0, 1, w)[None, :].repeat(h, axis=0)
            elif aug == 'top':
                grad = np.linspace(1, 0, h)[:, None].repeat(w, axis=1)
            else:
                grad = np.linspace(0, 1, h)[:, None].repeat(w, axis=1)
            ambient = 0.3
            light_map = ambient + grad * (1.0 - ambient)
            light_map = 1.0 - intensity * (1.0 - light_map)
            img_np = img_np * light_map[:, :, None]
        elif aug == 'overexpose':
            img_np = img_np + intensity * (255.0 - img_np)
        elif aug == 'underexpose':
            img_np = img_np * (1.0 - intensity)
        elif aug == 'tint':
            tint = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]) * intensity * 40
            img_np = img_np + tint[None, None, :]
        elif aug == 'warmth':
            warm = intensity * 25
            sign = random.choice([-1, 1])
            img_np[:, :, 0] += sign * warm       # R
            img_np[:, :, 2] -= sign * warm * 0.5  # B

        return np.clip(img_np, 0, 255)

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img_np = np.array(img).astype(np.float32)
        all_augs = ['left', 'right', 'top', 'bottom', 'overexpose', 'underexpose', 'tint', 'warmth']
        n_augs = random.choice([1, 2])
        chosen = random.sample(all_augs, n_augs)
        for aug in chosen:
            intensity = random.uniform(*self.intensity_range)
            img_np = self._apply_one(img_np, aug, intensity)
        img_np = img_np.astype(np.uint8)
        return Image.fromarray(img_np)


def get_data_transforms(size, isize, mean_train=None, std_train=None, lighting_aug=False):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    train_transforms_list = [transforms.Resize((size, size))]
    if lighting_aug:
        train_transforms_list.append(RandomLightingAugmentation(p=0.5))
    train_transforms_list.extend([
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train, std=std_train)])
    data_transforms = transforms.Compose(train_transforms_list)
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


def extract_tiles(img, overlap=0.5):
    """Extract 2x2 overlapping square tiles from a PIL Image.
    Returns: (list of 4 PIL crops, tile_info dict for stitching)"""
    w, h = img.size
    tile_h = int(h / (2 - overlap))
    tile_w = int(w / (2 - overlap))
    tile_side = min(tile_h, tile_w)

    stride_y = h - tile_side
    stride_x = w - tile_side

    tiles = []
    positions = [(0, 0), (0, stride_x), (stride_y, 0), (stride_y, stride_x)]
    for row, col in positions:
        tile = img.crop((col, row, col + tile_side, row + tile_side))
        tiles.append(tile)

    tile_info = {'h': h, 'w': w, 'tile_side': tile_side,
                 'stride_y': stride_y, 'stride_x': stride_x,
                 'positions': positions}
    return tiles, tile_info


class TiledImageFolder(torch.utils.data.Dataset):
    """Training dataset that yields individual tiles from each image."""
    def __init__(self, root, transform, overlap=0.5):
        self.transform = transform
        self.overlap = overlap
        self.samples = []
        for class_dir in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_dir)
            if not os.path.isdir(class_path):
                continue
            for ext in ('*.png', '*.JPG', '*.bmp'):
                for img_path in sorted(glob.glob(os.path.join(class_path, ext))):
                    self.samples.append(img_path)

    def __len__(self):
        return len(self.samples) * 4

    def __getitem__(self, idx):
        img_idx = idx // 4
        tile_idx = idx % 4
        img = Image.open(self.samples[img_idx]).convert('RGB')
        tiles, _ = extract_tiles(img, self.overlap)
        tile = tiles[tile_idx]
        return self.transform(tile), 0


class TiledMVTecDataset(torch.utils.data.Dataset):
    """Test dataset that yields tiles with GT tiles and metadata for stitching."""
    def __init__(self, root, transform, gt_transform, phase, overlap=0.5):
        self.img_path = os.path.join(root, 'test')
        self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        self.overlap = overlap
        self.img_paths, self.gt_paths, self.labels, self.types = self._load()

    def _load(self):
        img_tot, gt_tot, labels, types = [], [], [], []
        for defect_type in sorted(os.listdir(self.img_path)):
            imgs = sorted(glob.glob(os.path.join(self.img_path, defect_type, '*.png')) +
                         glob.glob(os.path.join(self.img_path, defect_type, '*.JPG')) +
                         glob.glob(os.path.join(self.img_path, defect_type, '*.bmp')))
            if defect_type == 'good':
                img_tot.extend(imgs)
                gt_tot.extend([0] * len(imgs))
                labels.extend([0] * len(imgs))
                types.extend(['good'] * len(imgs))
            else:
                gts = sorted(glob.glob(os.path.join(self.gt_path, defect_type, '*.png')))
                img_tot.extend(imgs)
                gt_tot.extend(gts)
                labels.extend([1] * len(imgs))
                types.extend([defect_type] * len(imgs))
        return np.array(img_tot), np.array(gt_tot), np.array(labels), np.array(types)

    def __len__(self):
        return len(self.img_paths) * 4

    def __getitem__(self, idx):
        img_idx = idx // 4
        tile_idx = idx % 4
        img_path = self.img_paths[img_idx]
        label = self.labels[img_idx]

        img = Image.open(img_path).convert('RGB')
        tiles, tile_info = extract_tiles(img, self.overlap)
        tile_img = self.transform(tiles[tile_idx])

        if label == 0:
            tile_gt = torch.zeros([1, tile_img.size(-2), tile_img.size(-1)])
        else:
            gt = Image.open(self.gt_paths[img_idx]).convert('L')
            gt_tiles, _ = extract_tiles(gt, self.overlap)
            tile_gt = self.gt_transform(gt_tiles[tile_idx])

        return tile_img, tile_gt, label, img_path, tile_idx, tile_info['h'], tile_info['w'], tile_info['tile_side'], tile_info['stride_y'], tile_info['stride_x']


class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, 'realiad_1024', category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types = [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, 'realiad_1024', category, sample['image_path']))
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, 'realiad_1024', category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path



