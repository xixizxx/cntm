# coding=utf-8
import sys
from src.evaluator import Evaluator


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
                args.load_from = sys.argv[3]
            evaluator = Evaluator(args)
            states = evaluator.run()
    else:
        print('Usage: "python evaluate.py Dataset [K] [Load_From]"')


if __name__ == '__main__':
    main()
