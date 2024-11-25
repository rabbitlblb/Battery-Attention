dataset=$1
fn=$2
ulimit -n 1048576
ulimit -s unlimited
python3 main_five_fold.py --config_path model_params_$1.json --fold_num $2
/bin/bash