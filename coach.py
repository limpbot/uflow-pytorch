import torch
import os
from clerk import Clerk
import numpy as np

from abc import ABCMeta

import usflow.optim
import usflow.data

# #### Setup options
from usflow import options
import sys

import pytorch_lightning as pl
from usflow.plusflow import PLUSFlow
from run_manager import RunManager

class Coach:
    __metaclass__ = ABCMeta

    def on_validation_epoch_end(self, trainer, pl_module: pl.LightningModule) -> None:
        print("validation epoch end")
        # if self.args.name_decisive_loss is not None:
        #    if self.val_metrics[self.args.name_decisive_loss] < self.val_loss_min:
        #        self.val_loss_min = self.val_metrics[self.args.name_decisive_loss]
        #        self.save_state(tag="best")

        print("save model")
        self.save_state()

    def __init__(self, args):
        print("")

        self.clerk = Clerk(args)
        self.args = self.clerk.args
        self.run_dir = self.clerk.run_dir

        # self.set_seed(self.args.train_seed)
        # Load Optimizer
        self.coach_state_dict_name = self.args.coach_state_dict_name

        pl.seed_everything(self.args.train_seed)
        # note: put model to cuda before otptimizer is initialized, because after .cuda() new object is generated
        self.model = PLUSFlow(args=self.args)

        if self.args.wandb_log_model:
            self.clerk.wandb_log_model(self.model)

        # self.val_epoch_length = 200 #len(self.val_dataloader)

        # self.val_steps = 0
        # self.train_losses_acc = {}
        # self.train_states_acc = {}
        # self.val_metrics_acc = {}
        # self.val_images_acc = {}

        # self.train_metrics = {}
        # self.val_metrics = {}

        # self.train_time_start = time.time()
        # self.val_time_start = time.time()

        # if torch.cuda.device_count() > 1:
        #    print("Let's use", torch.cuda.device_count(), "GPUs!")
        #    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #    self.model = torch.nn.DataParallel(self.model)
        # self.model.to(self.device)

    def run(self):

        self.args.dataloader_device = None  # self.device
        self.args.dataloader_pin_memory = False

        if self.args.dataloader_num_workers > 0:
            self.args.dataloader_device = "cpu"
            self.args.dataloader_pin_memory = True

        self.my_train_dataloader = usflow.data.create_train_dataloader(self.args)
        self.my_val_dataloader = usflow.data.create_val_dataloader(self.args)

        print("len train dataloader", len(self.my_train_dataloader))

        if self.args.train_epoch_length is not None:
            self.epoch_length = self.args.train_epoch_length
        else:
            self.epoch_length = len(self.my_train_dataloader)

        num_params = 0
        for param in self.model.parameters():
            num_params += param.numel()
        print("model has", num_params, "parameters")
        # self.model = self.model.cuda()

        self.optimizer = usflow.optim.load_optimizer(
            self.args.optimization,
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.args.lr_scheduler_steps,
            gamma=0.5,
            last_epoch=-1,
        )
        self.find_unused_parameters()

        self.load_state(tag=self.args.model_load_tag)

        self.model = self.model.cuda()

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        """
        for state in self.lr_scheduler.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                   state[k] = v.cuda()
        """
        self.model.optimizer = self.optimizer
        self.model.lr_scheduler = self.lr_scheduler
        self.model.clerk = self.clerk
        self.model.coach = self
        self.model.val_dataloader_length = len(self.my_val_dataloader)

        if self.args.debug_mode:
            if len(self.my_train_dataloader) * 0.01 > 1:
                self.args.train_limit_batches = 0.01
            if len(self.my_val_dataloader) * 0.01 > 1:
                self.args.val_limit_batches = 0.01

        pl_trainer = pl.Trainer(
            checkpoint_callback=False,
            max_epochs=self.args.train_num_epochs_max,
            gpus=-1,
            accelerator="ddp",
            accumulate_grad_batches=self.args.train_accumulate_grad_batches,
            precision=self.args.train_precision,
            sync_batchnorm=True,
            limit_train_batches=self.args.train_limit_batches,
            limit_val_batches=self.args.val_limit_batches,
            check_val_every_n_epoch=self.args.val_every_n_epoch,
            # global_step=self.optimizer.global_step
            # current_epoch=self.lr_scheduler.last_epoch,
        )

        pl_trainer.current_epoch = self.lr_scheduler.last_epoch
        pl_trainer.fit(self.model, self.my_train_dataloader, self.my_val_dataloader)

    def save_state(self, tag="latest"):

        self.clerk.save_model_state(self.model, tag)
        self.clerk.save_optimizer_state(self.optimizer, tag)
        self.clerk.save_lr_scheduler_state(self.lr_scheduler, tag)
        self.clerk.save_coach_state(self, tag)

    def load_state(self, tag="latest"):

        if self.args.run_start == "new":
            self.set_coach_state_default()
            if self.args.model_load_modules is not None:
                self.clerk.load_model_state(
                    self.model.architecture, tag, modules=self.args.model_load_modules
                )
        else:
            if self.args.run_start == "continue":
                self.clerk.load_optimizer_state(self.optimizer, tag)
                self.clerk.load_model_state(self.model.architecture, tag)

            elif self.args.run_start == "branch":
                self.clerk.load_optimizer_state(
                    self.optimizer, tag, modules=self.args.model_load_modules
                )
                self.clerk.load_model_state(
                    self.model.architecture, tag, modules=self.args.model_load_modules
                )
            else:
                print("error: unknown run_start passed", self.args.run_start)

            self.clerk.load_lr_scheduler_state(self.lr_scheduler, tag)
            self.clerk.load_coach_state(self, tag)

    def get_coach_state_dict(self):
        coach_state_dict = {}
        coach_state_dict["epoch"] = self.epoch
        if self.args.name_decisive_loss is not None:
            coach_state_dict[self.args.name_decisive_loss] = self.val_loss_min

        return coach_state_dict

    def set_coach_state_dict(self, coach_state_dict):
        if "epoch" in coach_state_dict:
            self.epoch = coach_state_dict["epoch"]

        if self.args.name_decisive_loss in coach_state_dict:
            self.val_loss_min = coach_state_dict[self.args.name_decisive_loss]

    def set_coach_state_default(self):
        self.epoch = 0
        self.val_loss_min = np.inf

    def find_unused_parameters(self):
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(self.my_train_dataloader):
            for i in range(len(batch)):
                batch[i] = batch[i][:1].to(self.model.device)
            losses, states = self.model.loss_train.calc_losses(batch, batch_idx)
            losses["total"].backward()
            break

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad == None:
                    print(name, "None")
                    param.requires_grad = False
                else:
                    pass
                    """
                    if torch.sum(param.grad.isinf()) > 0:
                        print(name, "inf")
                    if torch.sum(param.grad.isnan()) > 0:
                        print(name, "nan")
                    if torch.sum(param.grad == 0.) > 0:
                        print(name, "zero")
                    """
        self.optimizer.zero_grad()


def main():
    print("test branch leo")
    print("torch version", torch.__version__)
    print("pytorch-lightning version: ", pl.__version__)
    print("environ PATH: ", os.getenv("PATH", "not found"))
    print("repro directory", os.path.dirname(os.path.realpath(__file__)))
    # self.find_unused_parameters = True

    repro_dir = None
    # repro_dir = os.path.dirname(os.path.realpath(__file__))
    if repro_dir is None:
        repro_dir = "."
    # os.chdir(repro_dir)

    parser = options.setup_comon_options()
    args = parser.parse_args()

    args.repro_dir = repro_dir

    # args.model_load_modules
    # args = args
    args.debug_mode = "_pydev_bundle.pydev_log" in sys.modules.keys()
    if args.debug_mode:
        print("DEBUG MODE")
        args.dataloader_num_workers = 0

    run_manager = RunManager(args)

    coach = Coach(run_manager.args)

    coach.run()


if __name__ == "__main__":
    main()
