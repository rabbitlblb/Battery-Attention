dataset=$1
declare -a dn=(0 1 2 3 4)
for fn in 0 1 2 3 4
do
  echo screen -dmS $1$fn /bin/bash -c 'CUDA_VISIBLE_DEVICES='${dn[$fn]}' ./run_bt.sh '$dataset' '$fn''
  screen -dmS $1$fn /bin/bash -c 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='${dn[$fn]}' ./run_bt.sh '$dataset' '$fn''
done
