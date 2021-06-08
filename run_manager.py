import wandb
import os
from shutil import copy
import datetime

from usflow import options


class RunManager:
    def __init__(self, args_passed):
        print("init run manager")

        if os.getenv("LOCAL_RANK") is not None:
            print("ddp: this is for sure a ddp subproccess")

        if os.getenv("RUN_ID") is not None and os.getenv("RUN_DIR") is not None:
            print("using existing run_id, run_dir")

            self.run_id = os.environ["RUN_ID"]
            self.run_dir = os.environ["RUN_DIR"]
        else:
            print("new run_id, run_dir")
            # run_start: a) 'new' b) 'continue' c) 'branch'
            self.run_dir, self.run_id = self.retrieve_run_dir_and_id_from_args(
                args_passed
            )

            os.environ["RUN_ID"] = self.run_id
            os.environ["RUN_DIR"] = self.run_dir

        print("self.run_dir, self.run_id", self.run_dir, self.run_id)

        runs_load_ids = []

        # ensure state available
        if (
            args_passed.run_start == "continue"
            or args_passed.run_start == "branch"
            or args_passed.run_start == "present"
        ):
            runs_load_ids.append(args_passed.run_id)

        if args_passed.run_start != "continue" and args_passed.run_start != "present":
            if args_passed.model_load_modules is not None:
                for (
                    run_id,
                    model_load_modules,
                ) in args_passed.model_load_modules.items():
                    runs_load_ids.append(run_id)

        self.ensure_runs_available(args_passed, runs_load_ids)

        if args_passed.run_start == "branch":
            self.branch_run_dir(
                os.path.join(
                    args_passed.parent_runs_dir,
                    args_passed.runs_dir,
                    args_passed.run_id,
                ),
                self.run_dir,
            )

        # init wandb
        if (
            args_passed.wandb_log_metrics
            or args_passed.wandb_log_state
            or args_passed.run_start == "continue"
            or args_passed.run_start == "present"
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

        # retrieve args
        if args_passed.run_start == "new" or args_passed.run_start == "branch":
            self.copy_configs_from_args_to_dir(args_passed, self.run_dir)
            self.args = args_passed
        elif args_passed.run_start == "present":
            self.args = self.retrieve_args_from_dir(self.run_dir, args_passed)
            self.args.debug_mode = args_passed.debug_mode
            self.args.dataloader_num_workers = args_passed.dataloader_num_workers
            self.args.repro_dir = args_passed.repro_dir
        else:
            self.args = self.retrieve_args_from_dir(self.run_dir, args_passed)
            self.args.debug_mode = args_passed.debug_mode
            self.args.dataloader_num_workers = args_passed.dataloader_num_workers
            self.args.repro_dir = args_passed.repro_dir

        self.args.run_dir = self.run_dir
        self.args.run_id = self.run_id

    def copy_configs_from_args_to_dir(self, args, run_dir):
        copy(args.config_coach, run_dir)
        if args.wandb_log_state:
            wandb.save(os.path.join(run_dir, args.config_coach.split("/")[-1]))
            # copy(args.config_coach, os.path.join(run_dir, "wandb"))
        if args.config_coach_experiment is not None:
            copy(args.config_coach_experiment, run_dir)
            if args.wandb_log_state:
                wandb.save(
                    os.path.join(run_dir, args.config_coach_experiment.split("/")[-1])
                )
                # copy(args.config_coach_experiment, os.path.join(run_dir, "wandb"))

    def retrieve_args_from_dir(self, run_dir, args_passed):
        parser = options.setup_comon_options()
        run_dir_list = os.listdir(run_dir)

        configs_coach_def_list = list(
            filter(lambda k: k.startswith("config_coach_def"), run_dir_list)
        )

        configs_coach_exp_list = list(
            filter(lambda k: k.startswith("config_coach_exp"), run_dir_list)
        )

        cmd_parse_args = [
            "-s",
            "config/config_setup_0.yaml",
            "-c",
            os.path.join(run_dir, configs_coach_def_list[0]),
        ]

        if len(configs_coach_exp_list) > 0:
            cmd_parse_args += [
                "-e",
                os.path.join(run_dir, configs_coach_exp_list[0]),
            ]

        args, _ = parser.parse_known_args(cmd_parse_args)

        ## these args should be retrieved from passing args not from loaded
        args.project_name = args_passed.project_name
        args.team_name = args_passed.team_name
        args.wandb_log_metrics = args_passed.wandb_log_metrics
        args.wandb_log_state = args_passed.wandb_log_state

        args.run_id = args_passed.run_id
        args.run_start = args_passed.run_start
        args.run_tag = args_passed.run_tag
        args.model_load_tag = args_passed.model_load_tag
        args.model_load_modules = args_passed.model_load_modules

        return args

    def retrieve_run_dir_and_id_from_args(self, args):

        runs_dir = os.path.join(args.parent_runs_dir, args.runs_dir)
        if not os.path.exists(runs_dir):
            os.mkdir(runs_dir)

        if args.run_start == "new" or args.run_start == "branch":
            local_runs_dirs = os.listdir(runs_dir)

            if args.wandb_log_metrics or args.wandb_log_state:
                api = wandb.Api()
                runs = api.runs(args.team_name + "/" + args.project_name)
                online_runs_dirs = [run.id for run in runs]
            else:
                online_runs_dirs = []

            runs_dirs = local_runs_dirs + online_runs_dirs

            run_date = datetime.datetime.now().strftime("%Y_%m_%d")
            runs_same_date_versions = [
                int(dir.split("v")[1].split("_")[0])
                for dir in runs_dirs
                if dir.startswith(run_date)
            ]
            if len(runs_same_date_versions) == 0:
                run_version = 1
            else:
                if os.getenv("LOCAL_RANK") == None:
                    run_version = max(runs_same_date_versions) + 1
                else:
                    run_version = max(runs_same_date_versions)

            run_id = run_date + "_v" + str(run_version)
            if args.run_tag is not None and args.run_tag != "None":
                run_id = run_id + "_" + args.run_tag
        else:
            run_id = args.run_id

        run_dir = os.path.join(runs_dir, run_id)
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)

        return run_dir, run_id

    def ensure_runs_available(self, args, runs_ids):
        runs_available = True
        for run_id in runs_ids:
            run_available = self.ensure_run_available(args, run_id)
            if run_available == False:
                print("warning: run not available", run_id)
                runs_available = False

        return runs_available

    def ensure_run_available(self, args, run_id):
        run_load_dir = os.path.join(args.parent_runs_dir, args.runs_dir, run_id)
        local_run_available = True
        online_run_available = True

        suffix = ""
        if args.model_load_tag is not None and args.model_load_tag != "None":
            suffix = "_" + args.model_load_tag

        fnames_start_req = []
        fnames_start_req.append(args.model_state_dict_name + suffix)
        fnames_start_req.append(args.optimizer_state_dict_name + suffix)
        fnames_start_req.append(args.lr_scheduler_state_dict_name + suffix)
        fnames_start_req.append(args.coach_state_dict_name + suffix)
        fnames_start_req.append("config_coach_def")

        if os.path.exists(run_load_dir):
            fnames_dir = os.listdir(run_load_dir)
        else:
            fnames_dir = []
            local_run_available = False

        for fname_start_req in fnames_start_req:
            fname_start_req_available = False
            for fname_dir in fnames_dir:
                if fname_dir.startswith(fname_start_req):
                    fname_start_req_available = True
                    break
            if not fname_start_req_available:
                local_run_available = False

        if not local_run_available:
            print("warning: run not locally available")
            # fnames_online = run.files()[0].name
            api = wandb.Api()
            runs = api.runs(args.team_name + "/" + args.project_name)
            run = None
            for r in runs:
                if r.id == run_id:
                    run = r

            if run is not None:
                fnames_online = [file.name for file in run.files()]
                for fname_start_req in fnames_start_req:
                    fname_start_req_available = False
                    for fname_online in fnames_online:
                        if fname_online.startswith(fname_start_req):
                            print("avail_online ", fname_online)
                            fname_start_req_available = True
                            break
                    if not fname_start_req_available:
                        online_run_available = False
            else:
                online_run_available = False

            if online_run_available:
                print("info: restoring from wandb", run_id)

                if not os.path.exists(run_load_dir):
                    os.mkdir(run_load_dir)

                # init wandb
                wandb.init(
                    reinit=True,
                    id=run_id,
                    resume="allow",
                    project=args.project_name,
                    entity=args.team_name,
                    dir=run_load_dir,
                )

                fnames_start = fnames_start_req
                fnames_start.append("config_coach_exp")
                for fname_start in fnames_start:
                    for fname_online in fnames_online:
                        if fname_online.startswith(fname_start):
                            print("online fname:", fname_online)
                            file = wandb.restore(fname_online)
                            if file is not None:
                                print("local fpath:", file.name)
                                copy(file.name, run_load_dir)
                            else:
                                print("error: file is None")
                print("info: online run available")
        else:
            print("info: local run available")
        run_available = local_run_available or online_run_available

        return run_available

    def branch_run_dir(self, source_run_dir, target_run_dir):
        print("info: copying from ", source_run_dir, " to ", target_run_dir)
        fnames = os.listdir(source_run_dir)

        # filter out directories
        fnames = [f for f in fnames if os.path.isfile(os.path.join(source_run_dir, f))]

        # filter out yaml config files
        fnames = [f for f in fnames if not f.endswith(".yaml")]

        for fname in fnames:
            print(
                "info: copying from ",
                os.path.join(source_run_dir, fname),
                " to ",
                target_run_dir,
            )
            copy(os.path.join(source_run_dir, fname), target_run_dir)
