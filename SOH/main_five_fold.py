import argparse
import json
import os
import sys
import adaptrain
import extract
import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch battery Example')
    parser.add_argument('--config_path', type=str,
                        default='model_params_test.json')
    parser.add_argument('--fold_num', type=int, default=0)
    parser.add_argument('--low_rank', type=int, default=-1)
    parser.add_argument('--nl', type=int, default=-1)
    parser.add_argument('--hd', type=int, default=-1)
    parser.add_argument('--adaption', type=bool, default=False)

    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=p_args)
    print("Loaded configs at %s" % args.config_path)
    if args.nl >= 0:
        args.num_layers = args.nl
    if args.hd >= 0:
        args.hidden_size = args.hd
    if args.low_rank >= 0:
        args.battery_info_size = [args.low_rank] * args.num_layers
    args.save_model_path += f'_{args.num_layers}_{args.hidden_size}_{args.battery_info_size}'
    print("args", args)

    # train and save feature
    tr = adaptrain.Train_adaption(args, fold_num=args.fold_num)
    print('train start............................')
    tr.main()
    print('train end............................')
    modelparams_path=tr.getmodelparams()
    current_path = tr.current_path
    del tr