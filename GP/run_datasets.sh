for dni in 0 1 2 3 4
do
  echo CUDA_VISIBLE_DEVICES=$dni screen -dmS GP$dni /bin/bash -c 'python gaussian_process.py --fold_num '$dni''
  CUDA_VISIBLE_DEVICES=$dni screen -dmS GP$dni /bin/bash -c 'python gaussian_process.py --fold_num '$dni''
done