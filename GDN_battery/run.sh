for dni in 0 1 2 3 4
do
  echo screen -dmS GDN$dni /bin/bash -c './run_battery.sh '$dni' battery '$dni' 0 100'
  screen -dmS GDN$dni /bin/bash -c './run_battery.sh '$dni' battery '$dni' 0 100'
done
