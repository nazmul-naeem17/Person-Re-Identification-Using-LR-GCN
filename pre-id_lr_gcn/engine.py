from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim.sgd import SGD

# Absolute imports
from metrics import joint_scores, mAP
from model import PCB
from re_ranking import re_ranking
from utils import fliplr, l2_norm_standardize, plot_distributions

LOSSES = {
    'bce': F.binary_cross_entropy,
    'bce_logits': F.binary_cross_entropy_with_logits,
    'cross_entropy': F.cross_entropy,
    'nll_loss': F.nll_loss,
    'kl_div': F.kl_div,
    'mse': F.mse_loss,
    'l1_loss': F.l1_loss,
}

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise ValueError(f"Invalid boolean string value: {value}")

class ST_ReID(PCB, pl.LightningModule):
    def __init__(
        self,
        num_classes,
        learning_rate: float = 0.1,
        criterion: str = 'cross_entropy',
        rerank: bool = False,
        save_features: bool = True,
    ):
        super().__init__(num_classes)
        self.num_classes    = num_classes
        self.learning_rate  = learning_rate
        self.criterion      = LOSSES[criterion]
        self.rerank         = rerank
        self.save_features  = save_features

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-lp', '--log_path',       type=str,       default='./logs')
        parser.add_argument('-lr', '--learning_rate', type=float,     default=0.1)
        parser.add_argument('-c',  '--criterion',      choices=list(LOSSES.keys()), default='cross_entropy')
        parser.add_argument('-re', '--rerank',         type=str_to_bool, default=False)
        parser.add_argument('-sfe','--save_features', type=str_to_bool, default=False)
        parser.add_argument('-des','--description',    type=str,        default=None)
        parser.add_argument('--git-tag',               type=str_to_bool, default=False)
        parser.add_argument('--debug',                 type=str_to_bool, default=False)
        return parser

    def configure_optimizers(self):
        # 1) collect backbone & graph-conv parameters
        main_params = []
        main_params += list(self.feature_extractor.parameters())
        main_params += list(self.reduce_conv.parameters())
        for sg in self.sgconvs:
            main_params += list(sg.parameters())

        # 2) build parameter groups
        params_list = [{'params': main_params, 'lr': self.learning_rate}]
        # include each of the num_parts classifiers
        for m in range(self.num_parts):
            params_list.append({'params': getattr(self, f'classifier{m}').parameters()})

        optimizer = SGD(
            params_list,
            lr=self.learning_rate,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True
        )
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.130)
        return [optimizer], [scheduler]

    def prepare_data(self) -> None:
        # grab the ST distribution from the DataModule
        self.st_distribution = self.trainer.datamodule.st_distribution

    def shared_step(self, batch):
        x, y = batch
        if y.dtype != torch.long:
            y = y.long()

        parts = self(x, training=True)
        loss = 0
        preds_sum = torch.zeros_like(parts[0], dtype=torch.float32)
        for p in parts:
            loss += F.cross_entropy(p, y, label_smoothing=0.1)
            preds_sum += F.softmax(p, dim=1)
        preds = preds_sum.argmax(dim=1)
        acc = (preds == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log('Loss/train_loss',   loss, prog_bar=True)
        self.log('Accuracy/train_acc', acc,  prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        # reset feature buffers
        self.q_features = torch.Tensor()
        self.q_targets  = torch.Tensor()
        self.q_cam_ids  = torch.Tensor()
        self.q_frames   = torch.Tensor()
        self.g_features = torch.Tensor()
        self.g_targets  = torch.Tensor()
        self.g_cam_ids  = torch.Tensor()
        self.g_frames   = torch.Tensor()

    def eval_shared_step(self, batch, dataloader_idx):
        x, y, cam_ids, frames = batch
        feature_sum = None
        for _ in range(2):
            feat = self(x, training=False).detach().cpu().flatten(1)
            feature_sum = feat if feature_sum is None else feature_sum + feat
            x = fliplr(x, x.device)

        if dataloader_idx == 0:
            self.q_features = torch.cat([self.q_features, feature_sum])
            self.q_targets  = torch.cat([self.q_targets,  y.cpu()])
            self.q_cam_ids  = torch.cat([self.q_cam_ids,  cam_ids.cpu()])
            self.q_frames   = torch.cat([self.q_frames,   frames.cpu()])
        else:
            self.g_features = torch.cat([self.g_features, feature_sum])
            self.g_targets  = torch.cat([self.g_targets,  y.cpu()])
            self.g_cam_ids  = torch.cat([self.g_cam_ids,  cam_ids.cpu()])
            self.g_frames   = torch.cat([self.g_frames,   frames.cpu()])

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 2:
            loss, acc = self.shared_step(batch)
            self.log('Loss/val_loss',    loss, prog_bar=True)
            self.log('Accuracy/val_acc', acc,  prog_bar=True)
        else:
            self.eval_shared_step(batch, dataloader_idx)

    def on_validation_epoch_end(self) -> None:
        mean_ap, cmc = self._compute_metrics()
        self.log('Results/val_mAP',      mean_ap)
        self.log('Results/val_CMC_top1', cmc[0].item())
        self.log('Results/val_CMC_top5', cmc[4].item())

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx, dataloader_idx):
        self.validation_step(batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self) -> None:
        fig = plot_distributions(self.trainer.datamodule.st_distribution)
        self.logger.experiment.add_figure('Spatial-Temporal Distribution', fig)
        mean_ap, cmc = self._compute_metrics()
        self.log('Results/test_mAP',      mean_ap)
        self.log('Results/test_CMC_top1', cmc[0].item())
        self.log('Results/test_CMC_top5', cmc[4].item())

    def _compute_metrics(self):
        qf = l2_norm_standardize(self.q_features)
        gf = l2_norm_standardize(self.g_features)
        scores = joint_scores(
            qf, self.q_cam_ids, self.q_frames,
            gf, self.g_cam_ids, self.g_frames,
            self.trainer.datamodule.st_distribution
        )
        if self.rerank:
            scores = re_ranking(scores)
        return mAP(scores, self.q_targets, self.q_cam_ids, self.g_targets, self.g_cam_ids)
