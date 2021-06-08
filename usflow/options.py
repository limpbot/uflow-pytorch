import configargparse
import argparse
import yaml

'''
import os
from collections import OrderedDict
import yaml
import json

class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def construct_include(loader: Loader, node: yaml.Node):
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

yaml.add_constructor('!include', construct_include, Loader)

class MyYAMLConfigFileParser(configargparse.YAMLConfigFileParser):

    def parse(self, stream):
        """Parses the keys and values from a config file."""
        yaml = self._load_yaml()

        try:
            parsed_obj = yaml.load(stream, Loader)
        except Exception as e:
            raise configargparse.ConfigFileParserException("Couldn't parse config file: %s" % e)

        if not isinstance(parsed_obj, dict):
            raise configargparse.ConfigFileParserException("The config file doesn't appear to "
                "contain 'key: value' pairs (aka. a YAML mapping). "
                "yaml.load('%s') returned type '%s' instead of 'dict'." % (
                getattr(stream, 'name', 'stream'),  type(parsed_obj).__name__))

        result = OrderedDict()
        for key, value in parsed_obj.items():
            if isinstance(value, list):
                result[key] = value
            elif value is None:
                pass
            else:
                result[key] = str(value)

        return result

'''


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_comon_options():
    # Parse arguments
    parser = configargparse.ArgumentParser(
        description="UFlow Training",
        config_file_parser_class=configargparse.ConfigparserConfigFileParser,
    )
    # , config_file_parser_class=MyYAMLConfigFileParser)

    # Dataset options
    parser.add_argument(
        "-s",
        "--config-setup",
        required=True,
        is_config_file=True,
        help="Path to config file for setup parameters",
    )

    parser.add_argument(
        "-c",
        "--config-coach",
        required=True,
        is_config_file=True,
        help="Path to config file for coach parameters",
    )

    parser.add_argument(
        "-e",
        "--config-coach-experiment",
        required=False,
        is_config_file=True,
        default=None,
        help="Path to config file for experiment specific coach parameters",
    )

    parser.add_argument(
        "-d",
        "--datasets-dir",
        default=None,
        required=True,
        type=str,
        metavar="DIRS",
        help="path to datasets",
    )

    parser.add_argument(
        "--project-name",
        default="opticalflow",
        type=str,
        help="project-name, relevant for wandb",
    )

    parser.add_argument(
        "--team-name", default="oranges", type=str, help="team-name, relevant for wandb"
    )

    parser.add_argument(
        "--wandb-log-metrics",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--wandb-log-metrics, enable to log to wandb (default: False)",
    )

    parser.add_argument(
        "--wandb-log-state",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--wandb-log-state, enable to log to wandb (default: False)",
    )

    parser.add_argument(
        "--wandb-log-model",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--wandb-log-model, enable to log to wandb (default: False)",
    )

    parser.add_argument(
        "--parent-runs-dir",
        default="",
        type=str,
        help="possible option for other parent dir than working dir",
    )

    parser.add_argument(
        "--dataloader-num-workers",
        default=0,
        type=int,
        help="dataloader-num-workers (default: 0)",
    )

    parser.add_argument(
        "--train-epoch-length",
        default=None,
        type=int,
        metavar="N",
        help="epoch length (default: 1000)",
    )

    # Model options
    parser.add_argument(
        "--run-start",
        default="new",
        type=str,
        help="select how to start run: new, continue, branch (default: new)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        type=str,
        metavar="TAG",
        help="the name of the folder to save the model in (default: none)",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        type=str,
        metavar="TAG",
        help="the name of the folder to save the model in (default: none)",
    )
    parser.add_argument(
        "--model-load-tag", default="latest", type=str, help="(default: latest)"
    )
    parser.add_argument(
        "--model-load-modules",
        type=yaml.safe_load,
        default=None,
        help="--model-load-modules (default: None)",
    )

    parser.add_argument(
        "--model-freeze-modules",
        type=yaml.safe_load,
        default=None,
        action="append",
        help="--model-load-modules (default: None)",
    )

    parser.add_argument(
        "--name-decisive-loss",
        default=None,
        type=str,
        metavar="TAG",
        help="the validation loss that decides if the model should be saved (default: none)",
    )

    # TRAIN
    parser.add_argument("--train-seed", default=25, type=int, help="(default: 25)")

    parser.add_argument(
        "--train-dataset-max-num-imgs",
        type=int,
        default=None,
        help="default: none - equals no limit",
    )

    parser.add_argument(
        "--val-dataset-max-num-imgs",
        default=None,
        type=int,
        help="default: none - equals no limit",
    )

    parser.add_argument(
        "--val-dataset-index-shift",
        default=None,
        type=int,
        help="default: none - equals no shift",
    )

    parser.add_argument(
        "--train-dataset-index-shift",
        default=None,
        type=int,
        help="default: none - equals no shift",
    )

    parser.add_argument(
        "--val-every-n-epoch",
        default=1,
        type=int,
        help="default: 1",
    )

    parser.add_argument(
        "--train-limit-batches",
        default=1.0,
        type=float,
        help=" default: 1.0 equals 100% -> no limit",
    )

    parser.add_argument(
        "--val-limit-batches",
        default=1.0,
        type=float,
        help=" default: 1.0 equals 100% -> no limit",
    )

    parser.add_argument("--train-precision", default=16, type=int, help="(default: 16)")

    parser.add_argument(
        "--train-accumulate-grad-batches", default=1, type=int, help="(default: 1)"
    )

    parser.add_argument("--train-batch-size", default=1, type=int, help="(default: 1)")

    # TEST
    parser.add_argument(
        "--test-sflow-via-disp-se3",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--test-sflow-via-disp-se3 (default: False)",
    )

    # LOSS
    parser.add_argument(
        "--loss-balance-sf-disp",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--loss-balance-sf-disp (default: False)",
    )

    parser.add_argument(
        "--loss-lvl-weights",
        default=[],
        type=float,
        action="append",
        help="--loss-lvl-weights (default: [])",
    )

    parser.add_argument(
        "--loss-disp-photo-lambda",
        default=1.0,
        type=float,
        help="loss-disp-photo-lambda  (default: 1.)",
    )

    parser.add_argument(
        "--loss-disp-smooth-lambda",
        default=0.1,
        type=float,
        help="loss-disp-smooth-lambda  (default: 0.1)",
    )
    parser.add_argument(
        "--loss-disp-smooth-order",
        default=2,
        type=int,
        help="loss-disp-smooth: order of smoothness  (default: 2)",
    )
    parser.add_argument(
        "--loss-disp-smooth-edgeweight",
        default=10.0,
        type=float,
        help="loss-disp-smooth: weight of edge smoothness  (default: 10.0)",
    )

    parser.add_argument(
        "--loss-disp-flow-cons3d-lambda",
        default=0.2,
        type=float,
        help="loss-flow-cons3d-lambda  (default: .2)",
    )
    parser.add_argument(
        "--loss-disp-flow-cons3d-type",
        default="smsf",
        type=str,
        help="smsf (b.o. backward warping) or chamfer (b.o. nearest neighbor) (default: smsf)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-corr3d-joint-lambda",
        default=0.0,
        type=float,
        help="loss-disp-se3-proj2oflow-corr3d-joint-lambda  (default: 0.0)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-corr3d-separate-lambda",
        default=0.0,
        type=float,
        help="loss-disp-se3-proj2oflow-corr3d-separate-lambda  (default: 0.0)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-corr3d-separate-use-mask-from-loss",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--loss-balance-sf-disp (default: False)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-corr3d-separate-mask-lambda",
        default=0.0,
        type=float,
        help="loss-disp-se3-proj2oflow-corr3d-separate-mask-lambda  (default: 0.0)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-cross3d-score-const-weight",
        default=1.0,
        type=float,
        help="(default: 1.0)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-cross3d-score-linear-weight",
        default=10.0,
        type=float,
        help="(default: 10.0)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-cross3d-score-exp-weight",
        default=10.0,
        type=float,
        help="(default: 10.0)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-cross3d-score-exp-slope",
        default=500.0,
        type=float,
        help="(default: 500.0)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-cross3d-outlier-slope",
        default=0.1,
        type=float,
        help="(default: 0.1)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-cross3d-outlier-min",
        default=0.1,
        type=float,
        help="(default: 0.1)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-cross3d-outlier-max",
        default=0.9,
        type=float,
        help="(default: 0.9)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-cross3d-min",
        default=0.001,
        type=float,
        help="(default: 0.001)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-cross3d-max",
        default=0.1,
        type=float,
        help="(default: 0.1)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-corr3d-mask-certainty-slope",
        default=500.0,
        type=float,
        help="loss-disp-se3-proj2oflow-corr3d-mask-certainty-slope  (default: 500.)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-corr3d-pxl-certainty-slope",
        default=100.0,
        type=float,
        help="loss-disp-se3-proj2oflow-corr3d-pxl-certainty-slope  (default: 100.)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-corr3d-separate-mask-uncertainty-slope",
        default=0.1,
        type=float,
        help="loss-disp-se3-proj2oflow-corr3d-separate-mask-uncertainty-slope  (default: 0.1)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-corr3d-separate-mask-uncertainty-offset",
        default=1.0,
        type=float,
        help="loss-disp-se3-proj2oflow-corr3d-separate-mask-uncertainty-offset  (default: 1.0)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-fwdbwd",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="--loss-disp-se3-proj2oflow-fwdbwd (default: True)",
    )

    parser.add_argument(
        "--loss-disp-se3-proj2oflow-level0-only",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--loss-disp-se3-proj2oflow-level0-only (default: False)",
    )

    parser.add_argument(
        "--loss-flow-photo-lambda",
        default=1.0,
        type=float,
        help="loss-flow-photo-lambda  (default: 1.)",
    )
    parser.add_argument(
        "--loss-photo-type",
        default="ssim",
        type=str,
        help="ssim or census (default: ssim)",
    )

    parser.add_argument(
        "--loss-flow-smooth-lambda",
        default=200,
        type=float,
        help="loss-flow-smooth-lambda  (default: 200)",
    )
    parser.add_argument(
        "--loss-flow-smooth-order",
        default=2,
        type=int,
        help="loss-flow-smooth: order of smoothness  (default: 2)",
    )
    parser.add_argument(
        "--loss-flow-smooth-edgeweight",
        default=10.0,
        type=float,
        help="loss-sflow-smooth: weight of edge smoothness  (default: 10.0)",
    )

    parser.add_argument(
        "--loss-mask-cons-oflow-lambda",
        default=0.0,
        type=float,
        help="loss-mask-cons-oflow-lambda  (default: 0.)",
    )

    parser.add_argument(
        "--loss-mask-cons-oflow-fwdbwd",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="--loss-mask-cons-oflow-fwdbwd (default: True)",
    )

    parser.add_argument(
        "--loss-se3-diversity-lambda",
        default=0.0,
        type=float,
        help="loss-se3-diversity-lambda  (default: 0.)",
    )

    parser.add_argument(
        "--loss-disp-se3-photo-lambda",
        default=1.0,
        type=float,
        help="loss-disp-se3-photo-lambda  (default: 1.)",
    )
    parser.add_argument(
        "--loss-disp-se3-photo-fwdbwd",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="--loss-disp-se3-photo-fwdbwd (default: True)",
    )

    parser.add_argument(
        "--loss-disp-se3-cons3d-lambda",
        default=0.2,
        type=float,
        help="loss-disp-se3-cons3d-lambda  (default: .2)",
    )
    parser.add_argument(
        "--loss-disp-se3-cons3d-type",
        default="smsf",
        type=str,
        help="smsf (b.o. backward warping) or chamfer (b.o. nearest neighbor) (default: smsf)",
    )
    parser.add_argument(
        "--loss-disp-se3-cons3d-fwdbwd",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="--loss-disp-se3-cons3d-fwdbwd (default: True)",
    )

    parser.add_argument(
        "--loss-disp-se3-cons-oflow-lambda",
        default=0.0,
        type=float,
        help="loss-disp-se3-cons-oflow-lambda  (default: .)",
    )
    parser.add_argument(
        "--loss-disp-se3-cons-oflow-type",
        default="smsf",
        type=str,
        help="smsf (b.o. backward warping) or chamfer (b.o. nearest neighbor) (default: smsf)",
    )
    parser.add_argument(
        "--loss-disp-se3-cons-oflow-fwdbwd",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="--loss-disp-se3-cons-oflow-fwdbwd (default: True)",
    )

    parser.add_argument(
        "--loss-mask-reg-nonzero-lambda",
        default=0.0,
        type=float,
        help="loss-mask-reg-nonzero-lambda  (default: 0.)",
    )

    parser.add_argument(
        "--loss-mask-smooth-lambda",
        default=0.1,
        type=float,
        help="loss-mask-smooth-lambda  (default: 0.1)",
    )
    parser.add_argument(
        "--loss-mask-smooth-order",
        default=2,
        type=int,
        help="loss-mask-smooth: order of smoothness  (default: 2)",
    )
    parser.add_argument(
        "--loss-mask-smooth-edgeweight",
        default=10.0,
        type=float,
        help="loss-mask-smooth: weight of edge smoothness  (default: 10.0)",
    )

    parser.add_argument(
        "--loss-nonzeromask-thresh-perc",
        default=0.5,
        type=float,
        help="--loss-nonzeromask-thresh-perc (default: 0.5)",
    )

    parser.add_argument(
        "--loss-masks-non-occlusion-binary",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--sflow-loss-balance-sf-disp (default: False)",
    )

    parser.add_argument(
        "--loss-smooth-type",
        default="uflow",
        type=str,
        help="impact on how to calc 1st gradient: uflow: -1|0|1 or smsf: -1|1  (default: uflow)",
    )

    parser.add_argument(
        "--loss-pts3d-norm-detach",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--sflow-pts3d-norm-detach (default: False)",
    )

    # ARCHITECTURE

    parser.add_argument(
        "--arch-res-width", default=640, type=int, help="arch-res-width (default: 640)"
    )
    parser.add_argument(
        "--arch-res-height",
        default=640,
        type=int,
        help="arch-res-height (default: 640)",
    )

    parser.add_argument(
        "--arch-leaky-relu-negative-slope",
        default=0.1,
        type=float,
        help="--arch-leaky-relu-negative-slope (default: 0.1)",
    )

    parser.add_argument(
        "--arch-modules-features-channels",
        default=[],
        type=int,
        action="append",
        help="--arch-modules-features-channels (default: [])",
    )
    parser.add_argument(
        "--arch-modules-features-convs-per-lvl",
        default=3,
        type=int,
        help="2 or 3 (default: 2)",
    )

    parser.add_argument(
        "--arch-cost-volume-normalize-features",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--arch-cost-volume-normalize-features (default: False)",
    )

    parser.add_argument(
        "--arch-detach-context-disp-flow-lvlwise",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--arch-detach-context-disp-flow-lvlwise (default: False)",
    )
    parser.add_argument(
        "--arch-detach-context-disp-flow-before-refinement",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--arch-detach-context-disp-flow-before-refinement (default: False)",
    )

    parser.add_argument(
        "--arch-context-out-channels", default=32, type=int, help="(default: 32)"
    )
    parser.add_argument(
        "--arch-context-dropout",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--arch-context-dropout (default: False)",
    )
    parser.add_argument(
        "--arch-context-densenet",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--sflow-context-densenet (default: False)",
    )

    parser.add_argument(
        "--arch-flow-out-channels", default=3, type=int, help="(default: 3)"
    )
    parser.add_argument(
        "--arch-flow-res",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--arch-flow-res (default: False)",
    )
    parser.add_argument(
        "--arch-flow-dropout",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--arch-flow-dropout (default: False)",
    )
    parser.add_argument(
        "--arch-flow-encoder-type",
        default="pwcnet",
        type=str,
        help="--arch-flow-sep-encoder e.g. pwcnet, resnet (default: pwcnet)",
    )

    parser.add_argument(
        "--arch-disp-out-channels", default=1, type=int, help="(default: 1)"
    )

    parser.add_argument(
        "--arch-disp-encoder-channels",
        default=[],
        type=int,
        action="append",
        help="--arch-disp-encoder-channels (default: [])",
    )
    parser.add_argument(
        "--arch-disp-encoder",
        default="resnet",
        type=str,
        help="--arch-disp-sep-encoder e.g. resnet, unet, none (default: resnet)",
    )
    parser.add_argument(
        "--arch-disp-dropout",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--arch-disp-dropout (default: False)",
    )
    parser.add_argument(
        "--arch-disp-res",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--arch-disp-res (default: False)",
    )
    parser.add_argument(
        "--arch-disp-separate",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="--arch-disp-separate (default: True)",
    )
    parser.add_argument(
        "--arch-disp-activation-before-refinement",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--arch-disp-activation-before-refinement (default: False)",
    )
    parser.add_argument(
        "--arch-disp-activation",
        default="sigmoid",
        type=str,
        help="--arch-disp-activation e.g. identity, sigmoid, relu (default: sigmoid)",
    )

    parser.add_argument(
        "--arch-disp-activation-sigmoid-max-val",
        default=0.3,
        type=float,
        help="--arch-disp-activation-sigmoid-max-val (default: 0.3)",
    )

    parser.add_argument(
        "--arch-disp-rel-inside-model",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--arch-disp-rel-inside-model (default: False)",
    )

    parser.add_argument(
        "--arch-mask-out-init-bias",
        default="minus_ten",
        type=str,
        help="--arch-mask-out-init-bias e.g. minus_ten, zero (default: minus_ten)",
    )

    parser.add_argument(
        "--arch-mask-activation",
        default="sigmoid",
        type=str,
        help="--arch-mask-activation e.g. identity, sigmoid, relu, softmax, sigmoid+normalization (default: sigmoid)",
    )

    parser.add_argument(
        "--arch-module-se3-outs",
        default=6,
        type=int,
        help="arch-module-se3-outs (default: 6)",
    )
    parser.add_argument(
        "--arch-modules-masks-num-outs",
        default=1,
        type=int,
        help="arch-modules-masks-num-outs (default: 1)",
    )
    parser.add_argument(
        "--arch-module-se3-intermediate-channels",
        default=[],
        type=int,
        action="append",
        help="--arch-module-se3-intermediate-channels (default: [])",
    )
    parser.add_argument(
        "--arch-module-se3-intermediate-kernelsizes",
        default=[],
        type=int,
        action="append",
        help="--arch-module-se3-intermediate-kernelsizes (default: [])",
    )
    parser.add_argument(
        "--arch-module-se3-intermediate-strides",
        default=[],
        type=int,
        action="append",
        help="--arch-module-se3-intermediate-strides (default: [])",
    )
    parser.add_argument(
        "--arch-module-se3-input",
        default="imgpair",
        type=str,
        help="--arch-module-se3-input e.g. imgpair, features, context (default: imgpair)",
    )
    parser.add_argument(
        "--arch-module-se3-resnet-encoder",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--arch-module-se3-resnet-encoder (default: False)",
    )

    parser.add_argument(
        "--arch-se3-egomotion-addition",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="--arch-se3-egomotion-separate (default: True)",
    )

    parser.add_argument(
        "--arch-se3-encoded-cat-oflow",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="--arch-se3-encoded-cat-oflow (default: True)",
    )

    parser.add_argument(
        "--arch-se3-translation-sensitivity",
        default=0.01,
        type=float,
        help="(default: 0.01)",
    )

    parser.add_argument(
        "--arch-se3-rotation-sensitivity",
        default=0.001,
        type=float,
        help="(default: 0.001)",
    )

    """
    arch-se3-translation-sensitivity: 0.01
    arch-se3-rotation-sensitivity: 0.001
    """

    # Training options
    parser.add_argument(
        "--train-visualize",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="visualize training procedure (default: False)",
    )

    parser.add_argument(
        "--train-dataset-name",
        default="kitti-multiview",
        type=str,
        help="--train-dataset-name, e.g. kitti-multiview (default), kitti-raw-monosf-train",
    )

    # Training options
    parser.add_argument(
        "--eval-forward-backward",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="!!fill this out!! (default: False)",
    )
    parser.add_argument(
        "--train-with-occlusion-masks",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="!!fill this out!! (default: False)",
    )
    parser.add_argument("--occlusion-masks-type", default="wang", help="e.g. wang|brox")
    parser.add_argument(
        "--occlusion-brox-begin-epoch", default=50, type=int, help="e.g. 0, 50, 100"
    )

    parser.add_argument(
        "--train-with-selfsup",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="!!fill this out!! (default: False)",
    )

    parser.add_argument(
        "--loss-flow-teacher-crop-lambda",
        default=0.0,
        type=float,
        help="loss-flow-teacher-crop-lambda  (default: 0.)",
    )

    parser.add_argument(
        "--loss-flow-teacher-crop-begin-epoch",
        default=150,
        type=int,
        help="e.g. 100, 200, 500 (default: 150)",
    )

    parser.add_argument(
        "--loss-flow-teacher-crop-reduction-size",
        default=64,
        type=int,
        help="(default: 64)",
    )

    parser.add_argument(
        "--loss-flow-teacher-crop-rampup-epochs",
        default=100,
        type=int,
        help="(default: 100)",
    )
    """
    loss-flow-teacher-crop-lambda: 0.0
    loss-flow-teacher-crop-begin-epoch: 150
    loss-flow-teacher-crop-reduction-size: 64
    loss-flow-teacher-crop-rampup-epochs: 100
    """

    parser.add_argument(
        "--train-num-epochs-max",
        default=62,
        type=int,
        help="train-num-epochs-max  (default: 62)",
    )

    # Optimization options
    parser.add_argument(
        "-o",
        "--optimization",
        default="adam",
        type=str,
        metavar="OPTIM",
        help="type of optimization: sgd | [adam]",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        metavar="LR",
        help="initial learning rate (default: 1e-4)",
    )

    parser.add_argument(
        "--lr-scheduler-steps",
        default=[],
        type=float,
        action="append",
        help="--lr-scheduler-steps (default: [])",
    )

    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=0.0,
        type=float,
        metavar="W",
        help="weight decay (default: 0)",
    )

    # Display/Save options
    parser.add_argument(
        "--runs-dir",
        default="runs",
        type=str,
        metavar="ModelPATH",
        help="directory to save models in. If it doesnt exist, will be created. (default: models/)",
    )
    parser.add_argument(
        "--model-state-dict-name",
        default="model_state_dict",
        type=str,
        metavar="ModelSTATEDICT",
        help="directory of model state dict. (default: model_state_dict)",
    )
    parser.add_argument(
        "--optimizer-state-dict-name",
        default="optimizer_state_dict",
        type=str,
        metavar="OPTIMIZERSTATEDICT",
        help="directory of optimizer state dict. (default: optimizer_state_dict_name)",
    )
    parser.add_argument(
        "--lr-scheduler-state-dict-name",
        default="lr_scheduler_state_dict",
        type=str,
        metavar="LRSCHEDULERSTATEDICT",
        help="directory of optimizer state dict. (default: lr_scheduler_state_dict_name)",
    )
    parser.add_argument(
        "--coach-state-dict-name",
        default="coach_state_dict",
        type=str,
        metavar="COACHSTATEDICT",
        help="directory of codel state dict. (default: coach_state_dict)",
    )
    parser.add_argument(
        "--eval-save-visualizations",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="!!fill this out!! (default: False)",
    )

    # Return
    return parser
