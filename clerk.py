import json
import torch
import torchvision
import wandb
from tensorboardX import SummaryWriter
import cv2
from tensor_operations import vision as ops_vis
import os

import pytorch_lightning as pl


class Clerk:
    def __init__(self, args_passed):
        super().__init__()

        print("init clerk", os.getenv("LOCAL_RANK"))

        self.run_dir = args_passed.run_dir
        self.run_id = args_passed.run_id
        self.args = args_passed

        # init wandb
        """
        if (
                args_passed.wandb_log_metrics
                or args_passed.wandb_log_state
                or args_passed.run_start == "continue"
                or args_passed.run_start == "branch"
        ):
            wandb.init(
                reinit=True,
                id=self.run_id,
                resume="allow",
                project=args_passed.project_name,
                entity=args_passed.team_name,
                dir=self.run_dir,
            )
        """

        self.images_dir = os.path.join(self.run_dir, "images")
        if not os.path.exists(self.images_dir):
            os.mkdir(self.images_dir)

        self.eval_flows_dir = os.path.join(self.run_dir, "eval_flows")
        if not os.path.exists(self.eval_flows_dir):
            os.mkdir(self.eval_flows_dir)

        self.eval_depths_dir = os.path.join(self.run_dir, "eval_depths")
        if not os.path.exists(self.eval_depths_dir):
            os.mkdir(self.eval_depths_dir)

        self.eval_masks_dir = os.path.join(self.run_dir, "eval_masks")
        if not os.path.exists(self.eval_masks_dir):
            os.mkdir(self.eval_masks_dir)

        self.metrics = {}

        if args_passed.wandb_log_state:
            wandb.config.update(self.args, allow_val_change=True)

        self.tb_log_dir = os.path.join(self.run_dir, "tb_log")
        if not os.path.exists(self.tb_log_dir):
            os.mkdir(self.tb_log_dir)
        self.tb_writer = SummaryWriter(os.path.join(self.tb_log_dir))

        self.tb_log_config(vars(self.args))

        print("tensoboard logdir:", self.tb_log_dir)

    def tb_log_config(self, dict_config):
        text = ""
        for key, value in dict_config.items():
            num_spaces = 50 - len(key)
            if num_spaces < 0:
                num_spaces = 0
            text += key + " " + (" ." * num_spaces) + str(value) + " \n\n"
            # writer.add_text("config", key + " : " + str(value), walltime=0)

        self.tb_writer.add_text("config", text, walltime=0)

    def log_images(self, images, epoch):
        for key, val in images.items():
            file_path = os.path.join(self.images_dir, key.split("/")[-1] + ".png")
            cv2.imwrite(file_path, ops_vis.tensor_to_cv_img(val))
            # if key.startswith("Image"):
            images[key] = wandb.Image(val, caption=key)
        if self.args.wandb_log_metrics:
            self.wandb_log_metrics(images, epoch)

    def log_metrics(self, metrics, epoch, hparam_dict=None):
        # TODO: change step/epoch
        print("log_metrics", epoch, metrics)

        if self.args.wandb_log_metrics:
            self.wandb_log_metrics(metrics, epoch)

        tb_metrics = {}
        for key, val in metrics.items():
            if not key.startswith("Image"):
                tb_metrics[key] = metrics[key]

        self.tb_log_metrics(tb_metrics, epoch, hparam_dict=hparam_dict)

    def log_img(self, dir, name, img):
        file_path = os.path.join(dir, name)
        cv2.imwrite(file_path, ops_vis.tensor_to_cv_img(img))

        # wandb.save(file_path, base_path=self.run_dir)

    def wandb_log_model(self, model):
        # log: 'all', 'gradients', 'parameters', 'none' (default: 'gradients')
        # log_freq: number of optimization steps (default: 1000)
        wandb.watch(model)

    def wandb_log_metrics(self, metrics, epoch):
        wandb.log(metrics, step=epoch)
        # wandb.log({}, step=epoch + 1)

    def tb_log_metrics(self, metrics, epoch, hparam_dict=None):
        """Write an event to the tensorboard events file"""

        with torch.no_grad():
            for key, value in metrics.items():
                if value is not None:
                    self.tb_writer.add_scalar("{}".format(key), value, epoch)

    def tb_add_flow_prediction(self, flow_rgb, epoch):
        self.tb_writer.add_image("B-Flow/Predicted", flow_rgb, epoch)

    def tb_add_dataset_images(self, imgpairs, epoch):
        rgbdisp = torch.FloatTensor(2, 3, imgpairs.size(2), imgpairs.size(3))
        rgbdisp[0] = imgpairs[0, :3, :, :]
        rgbdisp[1] = imgpairs[0, 3:, :, :]
        rgbdisp = torchvision.utils.make_grid(rgbdisp)
        # rgbdisp   = rgbdisp[[2, 1, 0, 5, 4, 3],:,:]
        self.tb_writer.add_image("A-RGB/Inputs", rgbdisp, epoch)

    def save_dict_as_json(self, name, dict, tag=None):
        suffix = ".json"
        if tag is not None and tag != "None":
            suffix = "_" + tag + ".json"
        file_path = os.path.join(self.run_dir, name + suffix)
        with open(file_path, "w") as file:
            json.dump(dict, file, sort_keys=True, indent=4)

        if self.args.wandb_log_state:
            wandb.save(file_path)
            # file_path = os.path.join(self.run_dir, "wandb", name + suffix)
            # with open(file_path, 'w') as file:
            #    json.dump(dict, file, sort_keys=True, indent=4)

    def load_dict_from_json(self, name, tag=None):
        suffix = ".json"
        if tag is not None and tag != "None":
            suffix = "_" + tag + ".json"
        file_path = os.path.join(self.run_dir, name + suffix)
        with open(file_path, "r") as file:
            dict = json.load(file)
        return dict

    def save_torch(self, obj, name, tag=None):
        suffix = ".pt"
        if tag is not None and tag != "None":
            suffix = "_" + tag + ".pt"
        file_path = os.path.join(self.run_dir, name + suffix)

        torch.save(obj.state_dict(), file_path)
        if self.args.wandb_log_state:
            # file_path = os.path.join(self.run_dir, 'wandb', name + suffix)
            # torch.save(
            #    obj.state_dict(),
            #    file_path
            # )
            wandb.save(file_path)

    def load_torch(self, obj, name, tag=None, modules=None):
        suffix = ".pt"
        if tag is not None and tag != "None":
            suffix = "_" + tag + ".pt"

        state_dict = obj.state_dict()
        files_modules = {}

        if modules is not None:
            for run_id, file_modules in modules.items():
                file_path = os.path.join(
                    self.args.parent_runs_dir, self.args.runs_dir, run_id, name + suffix
                )
                files_modules[file_path] = file_modules
        else:
            file_path = os.path.join(self.run_dir, name + suffix)
            files_modules[file_path] = []

        for file_path, file_modules in files_modules.items():

            pretrained_state_dict = torch.load(file_path)
            # model_mapping/uflow_old_new.txt
            # my_io.read_mapping
            if "2020_10_26_uflow" in file_path:
                mapping = {}
                with open("model_mapping/uflow_old_new.txt", "r") as file:
                    for line in file.readlines():
                        line = line.rstrip("\n")
                        items = line.split(", ")
                        val = items[0]
                        key = items[1]
                        mapping[key] = val
                pretrained_state_dict_new = {}
                for key, val in pretrained_state_dict.items():
                    if key in mapping.keys():
                        pretrained_state_dict_new[mapping[key]] = pretrained_state_dict[
                            key
                        ]
                    else:
                        pretrained_state_dict_new[key] = pretrained_state_dict[key]
                pretrained_state_dict = pretrained_state_dict_new
            elif "2021_01_15_smsf" in file_path:
                pretrained_state_dict_new = {}
                for key, val in pretrained_state_dict.items():
                    if key.startswith("modules_features"):
                        pretrained_state_dict_new[
                            "module_flow.modules_features." + key
                        ] = val
                    else:
                        pretrained_state_dict_new["module_flow." + key] = val
                pretrained_state_dict = pretrained_state_dict_new

            pretrained_state_dict = {
                k: v for k, v in pretrained_state_dict.items() if k in state_dict
            }

            if name == "model_state_dict":
                if "module_se3.decoder.6.bias" in pretrained_state_dict.keys():
                    loaded_num_channels = pretrained_state_dict[
                        "module_se3.decoder.6.bias"
                    ].shape[0]
                    se3_dict = {}
                    se3_dict["module_se3.decoder.6.weight"] = state_dict[
                        "module_se3.decoder.6.weight"
                    ].clone()
                    se3_dict["module_se3.decoder.6.bias"] = state_dict[
                        "module_se3.decoder.6.bias"
                    ].clone()
                    se3_dict["module_se3.decoder.6.weight"][
                        :loaded_num_channels
                    ] = pretrained_state_dict["module_se3.decoder.6.weight"]
                    se3_dict["module_se3.decoder.6.bias"][
                        :loaded_num_channels
                    ] = pretrained_state_dict["module_se3.decoder.6.bias"]
                    pretrained_state_dict.update(se3_dict)
                    # pretrained_state_dict['module_se3.decoder.6.bias'] = se3_bias

            if file_modules is None or len(file_modules) == 0:
                pass
            else:
                pretrained_state_dict = {
                    k: v
                    for k, v in pretrained_state_dict.items()
                    if k.split(".")[0] in file_modules
                }

            # overwrite entries in the existing state dict
            state_dict.update(pretrained_state_dict)

        obj.load_state_dict(state_dict)

    def save_model_state(self, model, tag):
        self.save_torch(model, self.args.model_state_dict_name, tag)

    def load_model_state(self, model, tag, modules=None):
        self.load_torch(model, self.args.model_state_dict_name, tag, modules=modules)

    def save_optimizer_state(self, optimizer, tag):
        self.save_torch(optimizer, self.args.optimizer_state_dict_name, tag)

    def load_optimizer_state(self, optimizer, tag, modules=None):
        self.load_torch(
            optimizer, self.args.optimizer_state_dict_name, tag, modules=modules
        )

    def save_lr_scheduler_state(self, lr_scheduler, tag):
        self.save_torch(lr_scheduler, self.args.lr_scheduler_state_dict_name, tag)

    def load_lr_scheduler_state(self, lr_scheduler, tag):
        self.load_torch(lr_scheduler, self.args.lr_scheduler_state_dict_name, tag)

    def save_coach_state(self, coach, tag):
        coach_state_dict = coach.get_coach_state_dict()
        self.save_dict_as_json(self.args.coach_state_dict_name, coach_state_dict, tag)

    def load_coach_state(self, coach, tag):
        coach_state_dict = self.load_dict_from_json(
            self.args.coach_state_dict_name, tag
        )
        coach.set_coach_state_dict(coach_state_dict)

    @property
    def name(self):
        return ""

    @property
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    # def log_metrics(self, metrics, step):
    #    # metrics is a dictionary of metric names and values
    #    # your code to record metrics goes here
    #    pass

    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)

        pass
        # super().save()

    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
