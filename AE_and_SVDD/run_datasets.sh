dataset=$1
for fn in 0 1 2 3 4
do
  echo python traditional_methods.py --method $dataset --normalize --fold_num $fn
  python traditional_methods.py --method $dataset --normalize --fold_num $fn
done
/bin/bash
