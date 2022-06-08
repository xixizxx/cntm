# coding=utf-8
import os
import sys
import json5
from src.utils import params
from src.trainer import Trainer


def main():
    argv = sys.argv
    if len(argv) >= 2:
        dataset_name = sys.argv[1]
        config_file = 'configs/' + dataset_name + '.json5'
        arg_groups = params.parse(config_file)
        for args, config in arg_groups:
            if len(argv) >= 3:
                args.num_topics = int(sys.argv[2])
            if len(argv) >= 4:
                if sys.argv[3] == '0':
                    args.cl = 0
                    args.tau = 0
                else:
                    args.tau = float(sys.argv[3])
            trainer = Trainer(args)
            states = trainer.run()

    else:
        print('Usage: "python train.py Dataset [K] [Tau]"')


if __name__ == '__main__':
    main()
