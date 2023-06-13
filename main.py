import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import time
import torch
import numpy as np


torch.set_printoptions(sci_mode=False)

parser = argparse.ArgumentParser(description=globals()["__doc__"])

parser.add_argument(
    "--config", type=str, required=True, help="Path to the config file"
)
parser.add_argument('--device', type=int, default=0, help='GPU device id')
parser.add_argument('--thread', type=int, default=4, help='number of threads')
parser.add_argument("--seed", type=int, default=12321, help="Random seed")
parser.add_argument("--test_sample_seed", type=int, default=12321, help="Random seed during test time sampling")
parser.add_argument(
    "--exp", type=str, default="exp", help="Path for saving running related data."
)
parser.add_argument(
    "--doc",
    type=str,
    required=True,
    help="A string for documentation purpose. "
         "Will be the name of the log folder.",
)

parser.add_argument('--csv_file',
                    dest="csv_file",
                    help='csv file with information about the labels',
                    default="camelyon_csv_files/splits_0.csv", type=str)
parser.add_argument(
    "--dataroot", type=str, default=None,
    help="This argument will overwrite the dataroot in the config if it is not None."
)
parser.add_argument(
    "--comment", type=str, default="", help="A string for experiment comment"
)
parser.add_argument(
    "--verbose",
    type=str,
    default="info",
    help="Verbose level: info | debug | warning | critical",
)
parser.add_argument("--test", action="store_true", help="Whether to test the model")

parser.add_argument("--tune_T", 
                    action="store_true",
                    help="Whether to tune the scaling temperature parameter for calibration with training set.")

parser.add_argument("--sanity_check", 
                    action="store_true", 
                    help="Whether to quickly check test function implementation by running on only a few subsets.")


parser.add_argument(
    "--train_guidance_only",
    action="store_true",
    help="Whether to only pre-train the guidance classifier f_phi",
)
parser.add_argument(
    "--noise_prior",
    action="store_true",
    help="Whether to apply a noise prior distribution at timestep T",
)
parser.add_argument(
    "--no_cat_f_phi",
    action="store_true",
    help="Whether to not concatenate f_phi as part of eps_theta input",
)
parser.add_argument(
    "--add_ce_loss",
    action="store_true",
    help="Whether to add cross entropy loss",
)
parser.add_argument(
    "--eval_best",
    action="store_true",
    help="Evaluate best model during training, instead of the ckpt stored at the last epoch",
)
parser.add_argument("--fid", action="store_true")
parser.add_argument("--interpolation", action="store_true")
parser.add_argument(
    "--resume_training", action="store_true", help="Whether to resume training"
)

parser.add_argument(
    "--n_splits", type=int, default=10, help="total number of runs with different seeds for a specific task"
)
parser.add_argument(
    "--split", type=int, default=0, help="split ID"
)
parser.add_argument(
    "--ni",
    action="store_true",
    help="No interaction. Suitable for Slurm Job launcher",
)

parser.add_argument(
    "--skip_type",
    type=str,
    default="uniform",
    help="skip according to (uniform or quadratic)",
)
parser.add_argument(
    "--timesteps", type=int, default=None, help="number of steps involved"
)
parser.add_argument(
    "--eta",
    type=float,
    default=0.0,
    help="eta used to control the variances of sigma",
)
parser.add_argument("--sequence", action="store_true")

parser.add_argument(
    "--loss", type=str, default='ddpm', help="loss function"
)

parser.add_argument(
    "--num_sample", type=int, default=1, help="number of samples used in forward and reverse"
)


def parse_config(args):
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join(args.config), "r") as f:
        if args.test:
            config = yaml.unsafe_load(f)
            new_config = config
        else:
            config = yaml.safe_load(f)
            new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    if not args.ni:
        import torch.utils.tensorboard as tb

    # overwrite if dataroot is not None
    if not args.dataroot is None:
        new_config.data.dataroot = args.dataroot

    if not args.test :

        new_config.diffusion.noise_prior = True if args.noise_prior else False
        new_config.model.cat_y_pred = False if args.no_cat_f_phi else True
        if not args.resume_training:
            if not args.timesteps is None:
                new_config.diffusion.timesteps = args.timesteps
            if args.num_sample > 1:
                new_config.diffusion.num_sample = args.num_sample
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        if not args.ni:
            new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        else:
            new_config.tb_logger = None
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:

        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        # saving test metrics to a .txt file
        handler2 = logging.FileHandler(os.path.join(args.log_path, "testmetrics.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)



    # add device
    device_name = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed

    def seed_torch(seed=7):
        import random
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch(args.seed)

    return new_config, logger


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args = parser.parse_args()

    config, logger = parse_config(args)
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    if args.loss == 'card_onehot_conditional':
        from CARD import Diffusion
    else:
        raise NotImplementedError("Invalid loss option")

    try:
        runner = Diffusion(args, config, device=config.device)
        start_time = time.time()
        procedure = None

        if args.test:
            y_majority_vote_accuracy_all_steps_list = runner.test()
            procedure = "Testing"
        else:
            runner.train()
            procedure = "Training"
        end_time = time.time()
        logging.info("\n{} procedure finished. It took {:.4f} minutes.\n\n\n".format(
            procedure, (end_time - start_time) / 60))

    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    args.doc = args.doc + "/split_" + str(args.split)

    if args.test:
        args.config = args.config + args.doc + "/config.yml"

    sys.exit(main())
