from gc import callbacks
import os
from collections import defaultdict

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pytorch_lightning
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader
import pdb
from pathlib import Path

import dataset
import model
import utils
from evaluation.metrics import *
from evaluation import segmentation
from loss import Loss
from opt import get_opts
from utils import *
from hash_encoding import HashEmbedder, Linear_HashEmbedder, SHEncoder


class ValEveryNSteps(pytorch_lightning.Callback):
    def __init__(self, every_n_step):
        self.every_n_step = every_n_step

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
            trainer.run_evaluation()


class NeuralDiffSystem(pytorch_lightning.LightningModule):
    def __init__(self, hparams, train_dataset=None, val_dataset=None):
        super().__init__()
        self.hparams = hparams
        if self.hparams.deterministic:
            utils.set_deterministic()

        # for avoiding reinitialization of dataloaders when debugging/using notebook
        self.train_dataset = dataset.EPICDiff(vid=self.hparams.vid, root=self.hparams.root, split="train")
        self.val_dataset = dataset.EPICDiff(vid=self.hparams.vid, root=self.hparams.root, split="val")

        self.loss = Loss()

        self.models_to_train = []
        self.embeddings_to_train = []
        
        if not self.hparams.use_hash:
            self.embedding_xyz = model.PosEmbedding(
                hparams.N_emb_xyz - 1, hparams.N_emb_xyz
            )
            self.embedding_xyz_c = model.PosEmbedding(
                hparams.N_emb_xyz - 1, hparams.N_emb_xyz
            )
            self.embedding_dir = model.PosEmbedding(
                hparams.N_emb_dir - 1, hparams.N_emb_dir
            )
            xyz_dim = 6 * hparams.N_emb_xyz + 3
            dir_dim = 6 * hparams.N_emb_dir + 3
        else:
            # will set these in `setup` function
            self.embedding_xyz = HashEmbedder(bounding_box=self.train_dataset.bounding_box, \
                                              log2_hashmap_size=19, \
                                              finest_resolution=1024)
            self.embedding_xyz_c = HashEmbedder(bounding_box=self.train_dataset.bounding_box_c, \
                                              log2_hashmap_size=19, \
                                              finest_resolution=1024)
            self.embedding_dir = SHEncoder()
            xyz_dim = self.embedding_xyz.out_dim
            dir_dim = self.embedding_dir.out_dim
            self.embeddings_to_train += [self.embedding_xyz, self.embedding_xyz_c]

        self.embeddings = {
            "xyz": self.embedding_xyz,
            "xyz_c": self.embedding_xyz_c,
            "dir": self.embedding_dir,
        }

        self.embedding_t = model.LREEmbedding(
            N=hparams.N_vocab, D=hparams.N_tau, K=hparams.lowpass_K
        )
        self.embeddings["t"] = self.embedding_t
        self.models_to_train += [self.embedding_t]

        self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
        self.embeddings["a"] = self.embedding_a
        self.models_to_train += [self.embedding_a]

        if self.hparams.use_hash:
            self.nerf_coarse = model.NeuralDiff(
                "coarse",
                in_channels_xyz=xyz_dim,
                in_channels_dir=dir_dim,
                W=128,
                D=4,
                skips=[2],
            )
        else:
            self.nerf_coarse = model.NeuralDiff(
                "coarse",
                in_channels_xyz=xyz_dim,
                in_channels_dir=dir_dim,
                W=hparams.model_width,
            )
        self.models = {"coarse": self.nerf_coarse}
        if hparams.N_importance > 0:
            if self.hparams.use_hash:
                self.nerf_fine = model.NeuralDiff(
                    "fine",
                    in_channels_xyz=xyz_dim,
                    in_channels_dir=dir_dim,
                    encode_dynamic=True,
                    in_channels_a=hparams.N_a,
                    in_channels_t=hparams.N_tau,
                    beta_min=hparams.beta_min,
                    W=128,
                    D=4,
                    skips=[2],
                )
            else:
                self.nerf_fine = model.NeuralDiff(
                    "fine",
                    in_channels_xyz=xyz_dim,
                    in_channels_dir=dir_dim,
                    encode_dynamic=True,
                    in_channels_a=hparams.N_a,
                    in_channels_t=hparams.N_tau,
                    beta_min=hparams.beta_min,
                    W=hparams.model_width,
                )
            self.models["fine"] = self.nerf_fine
        self.models_to_train += [self.models]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, test_time=False, disable_perturb=False):
        perturb = 0 if test_time or disable_perturb else self.hparams.perturb
        noise_std = 0 if test_time or disable_perturb else self.hparams.noise_std
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = model.render_rays(
                models=self.models,
                embeddings=self.embeddings,
                rays=rays[i : i + self.hparams.chunk],
                ts=ts[i : i + self.hparams.chunk],
                N_samples=self.hparams.N_samples,
                perturb=perturb,
                noise_std=noise_std,
                N_importance=self.hparams.N_importance,
                chunk=self.hparams.chunk,
                hp=self.hparams,
                test_time=test_time,
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage, reset_dataset=False):
        kwargs = {"root": self.hparams.root}
        kwargs["vid"] = self.hparams.vid
        if (self.train_dataset is None and self.val_dataset is None) or reset_dataset:
            self.train_dataset = dataset.EPICDiff(split="train", **kwargs)
            self.val_dataset = dataset.EPICDiff(split="val", **kwargs)

    def configure_optimizers(self):
        self.optimizer = Adam([
            {'params': get_parameters(self.models_to_train), 'weight_decay': self.hparams.weight_decay},
            {'params': get_parameters(self.embeddings_to_train), 'eps': 1e-15}
            ], lr=self.hparams.lr
        )
        if not self.hparams.use_hash:
            scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.hparams.num_epochs, eta_min=1e-8
            )
            return [self.optimizer], [scheduler]
        else:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": ExponentialLR(self.optimizer, gamma=0.01**(1/50000), verbose=True),
                    "interval": "step",
                    "frequency": 1
                },
            }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        # batch_size=1 for validating one image (H*W rays) at a time
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            batch_size=1,
            pin_memory=True,
        )

    def training_step(self, batch, batch_nb):
        rays, rgbs, ts = batch["rays"], batch["rgbs"], batch["ts"]
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            psnr_ = psnr(results["rgb_fine"], rgbs)

        self.log("lr", self.optimizer.param_groups[0]["lr"])
        self.log("train/loss", loss)
        for k, v in loss_d.items():
            self.log(f"train/{k}", v, prog_bar=True)
        self.log("train/psnr", psnr_, prog_bar=True)

        return loss

    def render(self, sample, t=None, device=None):

        rays, rgbs, ts = (
            sample["rays"].cuda(),
            sample["rgbs"].cuda(),
            sample["ts"].cuda(),
        )

        if t is not None:
            if type(t) is torch.Tensor:
                t = t.cuda()
            ts = torch.ones_like(ts) * t

        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)
        with torch.no_grad():
            results = self(rays, ts, test_time=True)

        if device is not None:
            for k in results:
                results[k] = results[k].to(device)

        return results

    def validation_step(self, batch, batch_nb, is_debug=False):
        rays, rgbs, ts = batch["rays"], batch["rgbs"], batch["ts"]

        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)
        # disable perturb (used during training), but keep loss for tensorboard
        results = self(rays, ts, disable_perturb=True)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())
        log = {"val_loss": loss}

        WH = batch["img_wh"].view(1, 2)
        W, H = WH[0, 0].item(), WH[0, 1].item()
        img = (
            results["rgb_fine"].view(H, W, 3)[:, :, :3].permute(2, 0, 1).cpu()
        )  # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
        depth = visualize_depth(results["depth_fine"].view(H, W))  # (3, H, W)
        stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)

        if batch_nb == 0:
            if self.logger is not None:
                self.logger.experiment.add_images(
                    "val/GT_pred_depth", stack, self.global_step
                )

        psnr_ = psnr(results["rgb_fine"], rgbs)
        log["val_psnr"] = psnr_
        
        if is_debug:
            # then visualise in jupyter
            log["images"] = stack
            log["results"] = results

        if self.global_step>0:
            # f, p = plt.subplots(1, 3, figsize=(15, 15))
            # for i in range(3):
            #     im = stack[i]
            #     p[i].imshow(im.permute(1, 2, 0).cpu())
            #     p[i].axis("off")
            # # plt.show()
            results_seg = segmentation.evaluate_sample(self.val_dataset, batch_nb, t=ts[0], model=self, visualise=True, save=True)
            save_dir = os.path.join("ckpts/"+hparams.exp_name, "val_"+str(self.global_step))
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            # plt.savefig(os.path.join(save_dir, str(ts[0].item())+".png"), bbox_inches='tight')
            results_seg["figure"].savefig(os.path.join(save_dir, str(ts[0].item())+".png"), bbox_inches='tight')

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mean_psnr = torch.stack([x["val_psnr"] for x in outputs]).mean()

        self.log("val/loss", mean_loss)
        self.log("val/psnr", mean_psnr, prog_bar=True)


def init_trainer(hparams, logger=None, checkpoint_callback=None):
    if checkpoint_callback is None:
        checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
            filepath=os.path.join(f"ckpts/{hparams.exp_name}", "{epoch:d}"),
            monitor="val/psnr",
            mode="max",
            save_top_k=-1,
        )

    logger = pytorch_lightning.loggers.TestTubeLogger(
        save_dir="logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False,
        log_graph=False,
    )

    trainer = pytorch_lightning.Trainer(
        max_epochs=hparams.num_epochs,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=hparams.ckpt_path,
        logger=logger,
        weights_summary=None,
        progress_bar_refresh_rate=hparams.refresh_every,
        gpus=hparams.num_gpus,
        accelerator="ddp" if hparams.num_gpus > 1 else None,
        num_sanity_val_steps=1,
        benchmark=True,
        limit_train_batches=hparams.train_ratio,
        profiler="simple" if hparams.num_gpus == 1 else None,
        callbacks=[ValEveryNSteps(1000)] if hparams.use_hash else [],
        check_val_every_n_epoch=float('inf') if hparams.use_hash else 1
    )

    return trainer


def main(hparams):
    hparams.exp_name = hparams.vid+"/"
    if hparams.use_hash:
        hparams.exp_name += "hashenc_difflr"+str(hparams.lr)
    else:
        hparams.exp_name += "freq_lr"+str(hparams.lr)
    system = NeuralDiffSystem(hparams)
    trainer = init_trainer(hparams)
    trainer.fit(system)


if __name__ == "__main__":
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    hparams = get_opts()
    main(hparams)
