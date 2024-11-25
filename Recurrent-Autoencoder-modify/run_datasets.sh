dataset=$1
for fn in 0 1 2 3 4
do
  echo python main.py configs/config_lstm_ae_battery_$fn.json
  python main.py configs/config_lstm_ae_battery_$fn.json
done
/bin/bash
