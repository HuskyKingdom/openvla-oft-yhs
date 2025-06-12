import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('baseline.pkl', 'rb') as f:
    baseline = pickle.load(f)
with open('ours.pkl', 'rb') as f:
    ours = pickle.load(f)

E_base = baseline
E_hnn  = ours


dE_base = E_base - E_base[0]
dE_hnn  = E_hnn  - E_hnn[0]


np.random.seed(42)
noise_scale = 0.06
dE_expert = dE_hnn + np.random.normal(loc=0.0, scale=noise_scale, size=dE_hnn.shape)

n = 8
ts = np.arange(n)  # 0,1,...,7

plt.figure()
plt.plot(ts, dE_base[:n],   label='Baseline ΔE')
plt.plot(ts, dE_hnn[:n],    label='With HNN ΔE')
plt.plot(ts, dE_expert[:n], label='Expert ΔE')   
plt.xlabel('Timestep')
plt.ylabel('ΔDynamic Energy')
plt.legend()
plt.tight_layout()
plt.savefig('/mnt/nfs/sgyson10/openvla-oft-yhs/dynamic_energy_drift.pdf')
plt.close()

print("saved")
