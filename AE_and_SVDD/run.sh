declare -a dn=(auto_encoder deepsvdd)
for dni in 0 1 2 3 4
do
  echo CUDA_VISIBLE_DEVICES=$dni screen -dmS AE$dni /bin/bash -c 'python traditional_methods.py --method auto_encoder --normalize --fold_num '$dni''
  CUDA_VISIBLE_DEVICES=$dni screen -dmS AE$dni /bin/bash -c 'python traditional_methods.py --method auto_encoder --normalize --fold_num '$dni''
done