import os
import time

for d1 in os.listdir('dyad_vae_save'):
    if d1[:16]=='BatteryAttention':
        continue
    for d2 in os.listdir(f'dyad_vae_save/{d1}'):
        if not os.path.exists(f"dyad_vae_save/{d1}/{d2}/auc"):
            continue
        scores = []
        for f in os.listdir(f"dyad_vae_save/{d1}/{d2}/auc"):
            if f[:2]=='tf':
                scores.append(f[7:-4].split('-'))
        for s in scores:
            print(' '.join(d1.replace(' ','').split('_')[0:4]), d2, ' '.join([s[0], s[-1]]))
                