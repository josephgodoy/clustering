# extractor.py
# This script is for extracting max peak and mean noise tuples as pandas dataframes from
# astrosynth-generated synthetic pulsating variable star fourier transforms.

from astroSynth import PVS  #"Pulsating Variable Star"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
obj = PVS()

# Replace the string in "directory=" with your own path.

obj.load(directory='/home/joseph/SingleSine_50000')

mean_noise = list()
max_peak = list()
classification = list()

n_samples = int(input("How many samples do you want?"))

for freq, amp, classif, n in tqdm(obj.xget_ft(stop = n_samples), total = n_samples): #50000 in obj
    mean_noise.append(np.median(amp))
    max_peak.append(max(amp))
    classification.append(classif)
plt.plot(mean_noise, max_peak, 'o')
plt.show()

df = pd.DataFrame(data = {'meannoise':mean_noise, 'maxpeak':max_peak, 'classification':classification})
df = df.to_csv(str(n_samples) + '.csv')
