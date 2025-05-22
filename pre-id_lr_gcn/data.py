from pathlib import Path
from typing import Optional, Union
from typing_extensions import Literal

import joblib
import pytorch_lightning as pl
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

# Absolute imports
from metrics import smooth_st_distribution
from utils import get_ids


class ReIDDataset(Dataset):
    """
    Custom Dataset for Market & Duke Person Re-Identification.
    Returns (img, label) or (img, label, cam_id, frame) if ret_camid_n_frame=True.
    Skips corrupted images automatically.
    """
    def __init__(
        self,
        data_dir: str,
        dataset: Literal['market', 'duke'] = 'market',
        transform=None,
        target_transform=None,
        ret_camid_n_frame: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            raise FileNotFoundError(f"Invalid data directory: {self.data_dir}")
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.ret_camid_n_frame = ret_camid_n_frame
        self._init_data()

    def _init_data(self):
        # gather all .jpg files, verify they're valid images
        all_paths = list(self.data_dir.glob('*.jpg'))
        valid = []
        for p in all_paths:
            try:
                with Image.open(p) as img:
                    img.verify()
                valid.append(p)
            except (UnidentifiedImageError, OSError):
                print(f"Skipping corrupted image: {p}")
        # filter out any with '-1' in the filename
        self.imgs = [p for p in valid if '-1' not in p.stem]

        # parse ids
        self.cam_ids, self.labels, self.frames = get_ids(self.imgs, self.dataset)
        # build class→index map
        self.classes = tuple(sorted(set(self.labels)))
        self.class_to_idx = {lbl: i for i, lbl in enumerate(self.classes)}
        self.targets = [self.class_to_idx[lbl] for lbl in self.labels]
        self.num_cams = len(set(self.cam_ids))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        p = self.imgs[idx]
        img = Image.open(str(p)).convert('RGB')
        label = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        if self.ret_camid_n_frame:
            return img, label, self.cam_ids[idx], self.frames[idx]
        return img, label


class ReIDDataModule(pl.LightningDataModule):
    """
    LightningDataModule for Person ReID.
    - Splits train_dir into train/val (classification).
    - Uses query_dir & test_dir as query/gallery (retrieval).
    """
    def __init__(
        self,
        data_dir: str,
        dataset: Literal['market', 'duke'] = 'market',
        st_distribution: Optional[str] = None,
        train_subdir: str = 'bounding_box_train',
        test_subdir: str = 'bounding_box_test',
        query_subdir: str = 'query',
        train_batchsize: int = 16,
        val_batchsize: int = 16,
        test_batchsize: int = 16,
        num_workers: int = 2,
        random_erasing: float = 0.0,
        color_jitter: bool = False,
        save_distribution: Union[bool, str] = False,
    ):
        super().__init__()
        self.data_dir        = Path(data_dir)
        self.dataset         = dataset
        self.train_dir       = self.data_dir / train_subdir
        self.test_dir        = self.data_dir / test_subdir
        self.query_dir       = self.data_dir / query_subdir
        self.train_batchsize = train_batchsize
        self.val_batchsize   = val_batchsize
        self.test_batchsize  = test_batchsize
        self.num_workers     = num_workers
        self.random_erasing  = random_erasing
        self.color_jitter    = color_jitter
        self.st_distribution = st_distribution
        self.save_distribution = save_distribution
        self.prepare_data()

    @classmethod
    def from_argparse_args(cls, args):
        return cls(
            data_dir=args.data_dir,
            dataset=args.dataset,
            st_distribution=getattr(args, 'st_distribution', None),
            train_subdir=args.train_subdir,
            test_subdir=args.test_subdir,
            query_subdir=args.query_subdir,
            train_batchsize=args.train_batchsize,
            val_batchsize=args.val_batchsize,
            test_batchsize=args.test_batchsize,
            num_workers=args.num_workers,
            random_erasing=args.random_erasing,
            color_jitter=args.color_jitter,
            save_distribution=args.save_distribution,
        )

    def prepare_data(self):
        # Stronger augmentations for training
        train_tf = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.RandomCrop((384, 128), padding=0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02,0.33), ratio=(0.3,3.3), value='random'),
        ])

        # Simplified test transforms
        test_tf = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])

        # Full train dataset & split
        full_train = ReIDDataset(self.train_dir, self.dataset, transform=train_tf)
        self.num_classes = len(full_train.classes)
        n_train = int(0.8 * len(full_train))
        n_val   = len(full_train) - n_train
        self.train, self.val_split = random_split(full_train, [n_train, n_val])

        # Retrieval sets
        self.query   = ReIDDataset(self.query_dir, self.dataset, transform=test_tf, ret_camid_n_frame=True)
        self.gallery = ReIDDataset(self.test_dir,  self.dataset, transform=test_tf, ret_camid_n_frame=True)

        # ST distribution
        self._load_st_distribution()
        if self.save_distribution:
            self._save_st_distribution()

    def _load_st_distribution(self):
        if isinstance(self.st_distribution, str):
            p = Path(self.st_distribution)
            if not (p.exists() and p.is_file()):
                raise FileNotFoundError(f"st_distribution file not found: {p}")
            if p.suffix != '.pkl':
                raise ValueError("st_distribution must be a .pkl file")
            self.st_distribution = joblib.load(str(p))
        else:
            cam_ids = self.query.cam_ids + self.gallery.cam_ids
            targets = self.query.targets + self.gallery.targets
            frames  = self.query.frames  + self.gallery.frames
            num_cams = self.query.num_cams
            max_hist = 5000 if self.dataset=='market' else 3000
            self.st_distribution = smooth_st_distribution(cam_ids, targets, frames, num_cams, max_hist)

    def _save_st_distribution(self):
        out = self.save_distribution if isinstance(self.save_distribution, str) else str(self.data_dir/'st_distribution.pkl')
        if not out.endswith('.pkl'):
            out += '.pkl'
        joblib.dump(self.st_distribution, out)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.train_batchsize,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        query_loader = DataLoader(self.query,
                                  batch_size=self.test_batchsize,
                                  shuffle=False,
                                  num_workers=self.num_workers,
                                  pin_memory=True)
        gallery_loader = DataLoader(self.gallery,
                                    batch_size=self.test_batchsize,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    pin_memory=True)
        val_cls_loader = DataLoader(self.val_split,
                                    batch_size=self.val_batchsize,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    pin_memory=True)
        return [query_loader, gallery_loader, val_cls_loader]

    def test_dataloader(self):
        query_loader   = DataLoader(self.query,
                                    batch_size=self.test_batchsize,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    pin_memory=True)
        gallery_loader = DataLoader(self.gallery,
                                    batch_size=self.test_batchsize,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    pin_memory=True)
        return [query_loader, gallery_loader]
