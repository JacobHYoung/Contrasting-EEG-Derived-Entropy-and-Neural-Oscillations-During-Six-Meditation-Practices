# July 25, 2020. Ross Kempner
'''
If we use mne_features we must cite: Jean-Baptiste SCHIRATTI, Jean-Eudes LE DOUGET, Michel LE VAN QUYEN, Slim ESSID, Alexandre GRAMFORT, "An ensemble learning approach to detect epileptic seizures from long intracranial EEG recordings" Proc. IEEE ICASSP Conf. 2018.

Also, worth looking into this paper: Physiological time-series analysis using approximate entropy and sample entropy

Plan: Use at least three different python libraries calculation of approximate entropy to provide converging proof.

'''
# import various libraries  
import mne_features
import mne 
import entropy
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt


# lets create some fake data to test the various functions
fake_data = np.random.rand(10,1000)

# testing mne_features
print("approximate_entropy with mne_features: ",mne_features.univariate.compute_app_entropy(fake_data, emb=2, metric='chebyshev'))
mne_ap_en = mne_features.univariate.compute_app_entropy(fake_data, emb=2, metric='chebyshev')
# create a graph with the channels as the x column and the entropy as the y column
fig, ax = plt.subplots()
channel_list = []
for index, channel_row in enumerate(fake_data):
    channel_list.append(index)
ax.plot(channel_list,mne_ap_en)

ax.set(xlabel='Channels', ylabel='ApEn',
       title='Approximate Entropy of 10 Fake Data Channels Over 1000 Fake Time Points: ')
ax.grid()

fig.savefig("mne_features_ap_en_fake_data.png")
plt.show()

# testing EntroPy
EntroPy_app_entropy_list = []
for channel_row in fake_data:
    new_channel_entropy = entropy.app_entropy(channel_row, order=2, metric='chebyshev')
    EntroPy_app_entropy_list.append(new_channel_entropy)
print("approximate_entropy with EntroPy: ", EntroPy_app_entropy_list)
