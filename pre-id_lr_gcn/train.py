import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Absolute imports
from data import ReIDDataModule
from engine import ST_ReID
from utils import save_args


def train(args):
    # ------------------------------------------------------------------
    # 1) Set up log & checkpoint directories
    # ------------------------------------------------------------------
    log_path = Path(args.log_path) / 'reid_logs'
    version = 0
    while (log_path / f'version_{version}').exists():
        version += 1
    version_dir = log_path / f'version_{version}'
    os.makedirs(version_dir, exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir=version_dir,
        name="reid_logs"
    )

    checkpoint_dir = version_dir / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    chkpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='Results/val_mAP',
        save_last=True,
        mode='max',
        verbose=True,
        save_top_k=3
    )

    early_stop_callback = EarlyStopping(
        monitor="Results/val_mAP",
        patience=35,
        verbose=True,
        mode="max"
    )

    # ------------------------------------------------------------------
    # 2) Data & Model
    # ------------------------------------------------------------------
    data_module = ReIDDataModule.from_argparse_args(args)
    model       = ST_ReID(
        data_module.num_classes,
        learning_rate=args.learning_rate,
        criterion=args.criterion,
        rerank=args.rerank
    )

    save_args(args, version_dir)

    # ------------------------------------------------------------------
    # 3) Trainer (auto‐detect GPU/CPU)
    # ------------------------------------------------------------------
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"

    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        precision=32,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        logger=tb_logger,
        callbacks=[chkpt_callback, early_stop_callback],
        enable_progress_bar=True,
        num_sanity_val_steps=0
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = ST_ReID.add_model_specific_args(parser)
    parser = ReIDDataModule.add_argparse_args(parser)
    # you can add --gpus / --cpu flags here if you like

    args = parser.parse_args()
    print(f'\nArguments: \n{args}\n')
    train(args)
