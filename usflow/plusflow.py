import torch

import tensor_operations.vision as ops_vis
import pytorch_lightning as pl
import os

from usflow.loss_train import USFlowLossTrain
from usflow.loss_val import USFlowLossVal
from usflow.architecture import USFlow

class PLUSFlow(pl.LightningModule):
    def __init__(self, args):
        super(PLUSFlow, self).__init__()

        print("initialize model: fp_channels", args.arch_modules_features_channels)

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args
        if self.args.train_visualize:
            self.train_vwriter = ops_vis.create_vwriter(
                "optimization", self.args.arch_res_width, self.args.arch_res_height * 4
            )
        else:
            self.train_vwriter = None

        self.loss_train = USFlowLossTrain(self, args)
        self.loss_val = USFlowLossVal(self, args)
        # self.dtype = torch.float32

        self.architecture = USFlow(args)

        self.train_metrics = {}
        self.train_metrics_acc = {}
        self.train_num_batches_ddp = 0
        self.val_metrics = {}
        self.val_metrics_acc = {}
        self.val_num_batches_ddp = 0

    def forward(self, *args, **kwargs):
        return self.architecture.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        losses, state = self.loss_train.calc_losses(
            batch, batch_idx, self.train_vwriter
        )

        for key, val in state.items():
            self.log(
                key,
                val,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=False,
                logger=False,
            )

        self.train_num_batches_ddp += 1
        for key, val in losses.items():

            self.log(
                key,
                val,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
                logger=False,
            )
            key = "B-train/" + key

            if key not in self.train_metrics_acc:
                self.train_metrics_acc[key] = val.item()
            else:
                self.train_metrics_acc[key] += val.item()
        return losses["total"]

    def training_epoch_end(self, training_step_outputs):
        print("self.train_num_batches_ddp", self.train_num_batches_ddp)
        print("self.lr_scheduler.last_epoch", self.lr_scheduler.last_epoch)

        self.train_metrics = {}
        for key, val in self.train_metrics_acc.items():
            self.train_metrics[key] = val / self.train_num_batches_ddp

        # self.train_metrics["epoch"] = self.current_epoch
        self.clerk.log_metrics(
            self.train_metrics, epoch=self.current_epoch, hparam_dict=None
        )

        self.train_metrics_acc = {}
        self.train_num_batches_ddp = 0

    def validation_step(self, batch, batch_idx):
        world_size = os.getenv("WORLD_SIZE")
        local_rank = os.getenv("LOCAL_RANK")
        # print("local_rank", local_rank)
        # print("world_size", world_size)
        if (
            world_size is not None
            and world_size != "None"
            and local_rank is not None
            and local_rank != "None"
        ):
            world_size = int(world_size)
            local_rank = int(local_rank)
            prev_batch_idx = batch_idx
            # batch_idx = (
            #    int(self.val_dataloader_length / world_size) * local_rank + batch_idx
            # )
            batch_idx = int(batch_idx * world_size) + local_rank
            # print("updating batch_idx", prev_batch_idx, "->", batch_idx)

        metrics, images = self.loss_val.calc_losses(batch, batch_idx)

        eval_images = {}
        for key, val in images.items():
            new_key = "Images/" + key + str(batch_idx)
            eval_images[new_key] = images[key]
        self.clerk.log_images(eval_images, self.current_epoch)

        self.val_num_batches_ddp += 1
        for key, val in metrics.items():
            key = "A-eval/" + key

            if key not in self.val_metrics_acc:
                self.val_metrics_acc[key] = val.item()
            else:
                self.val_metrics_acc[key] += val.item()
            # self.log(key, val, prog_bar=True, on_step=True, on_epoch=False, sync_dist=False, logger=False)

        # print(self._results)
        # print(self.get_progress_bar_dict())
        return metrics[self.args.name_decisive_loss.split("/")[1]]

    def validation_epoch_end(self, validation_step_outputs):
        print("self.val_num_batches_ddp", self.val_num_batches_ddp)
        print("self.lr_scheduler.last_epoch", self.lr_scheduler.last_epoch)
        self.val_metrics = {}
        for key, val in self.val_metrics_acc.items():
            self.val_metrics[key] = val / self.val_num_batches_ddp

        self.val_metrics_acc = {}
        self.val_num_batches_ddp = 0

        # self.val_metrics["epoch"] = self.current_epoch
        self.clerk.log_metrics(
            self.val_metrics, epoch=self.current_epoch, hparam_dict=None
        )

        if (
            os.getenv("LOCAL_RANK") == "0"
            or os.getenv("LOCAL_RANK") == None
            or os.getenv("LOCAL_RANK") == "None"
        ):
            print("save model")
            self.coach.save_state()

    def configure_optimizers(self):
        return [self.optimizer,], [
            self.lr_scheduler,
        ]

    def __del__(self):
        if self.train_vwriter is not None:
            self.train_vwriter.release()
