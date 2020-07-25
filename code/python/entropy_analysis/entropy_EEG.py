# July 25, 2020. Ross Kempner
'''
If we use mne_features we must cite: Jean-Baptiste SCHIRATTI, Jean-Eudes LE DOUGET, Michel LE VAN QUYEN, Slim ESSID, Alexandre GRAMFORT, "An ensemble learning approach to detect epileptic seizures from long intracranial EEG recordings" Proc. IEEE ICASSP Conf. 2018.

Also, worth looking into this paper: Physiological time-series analysis using approximate entropy and sample entropy

Plan: Use at least three different python libraries calculation of approximate entropy to provide converging proof.

'''
# import various libraries  
import mne_features
import mne 
import numpy as np

# lets create some fake data to test the various functions. Mne_features has a function for computing the approximate entropy that 