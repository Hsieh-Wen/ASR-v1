
import sys
from config_argparse import ArgumentParser


def load_config(config_path=None):
    """load audio splitter config,
        priority:
        1. sys cmd
        2. config_path
        3. default

    Args:
        config_path (str, optional): config path, 
        if None, default is split_config.yaml.

    Returns:
        [args]: [argparser namespace]
    """

    parser = ArgumentParser()
    parser.parse_known_args()
    
    if len(sys.argv) > 1:
        # if have sys cmd will use sys.argv
        args = parser.parse_args()
        print(f"config_path:{sys.argv[1:]}")
        return args

    # if config_path is None:
    #     # if sys.argv and config_path is not exist, set default
    #     config_path = "./conf/split_config.yaml"

    cmd = [
        "--config",
        config_path,
    ]
    print(f"config_path:{config_path}")
    args = parser.parse_args(cmd)
    
    return args