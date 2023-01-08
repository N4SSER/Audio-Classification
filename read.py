import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
from librosa import load
def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1



def envelope(x,Fs,t):
    mask = []
    x = pd.Series(x).apply(np.abs)
    x_mean = x.rolling(window = int(Fs/10),min_periods=1,center = True).mean()
    for mean in x_mean:
        if mean > t:
            mask.append(True)
        else:
            mask.append(False)

    return mask

df = pd.read_csv('instruments.csv')
df.set_index('fname',inplace=True)

for f in df.index:
    sample_rate, x = wavfile.read('wavfiles/'+f)
    df.at[f,'length'] = x.shape[0]/sample_rate

classes = list(np.unique(df.label))

class_dist = df.groupby(['label'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution')
ax.pie(class_dist,labels=class_dist.index, autopct = '%1.1f%%',shadow=False,startangle=90)
ax.axis('equal')
#plt.show()
df.reset_index(inplace=True)
signals={}
fft = {}
fbank = {}
mfccs ={}

def _fft(x,Fs):
    N = len(x)
    freqs = np.fft.rfftfreq(len(x),d=1/sample_rate)
    X = abs(np.fft.rfft(x)/N)
    return (X,freqs)

for c in classes:
    wav_file = df[df.label == c].iloc[0,0]
    x,sample_rate = load('wavfiles/'+wav_file,sr=44100)
    mask = envelope(x,sample_rate,0.0005)
    x = x[mask]
    signals[c] = x
    fft[c] = _fft(x,sample_rate)
    bank = logfbank(x[:sample_rate],sample_rate,nfilt=26,nfft = 1103).T
    fbank[c] = bank
    mel = mfcc(x[:sample_rate],sample_rate,numcep=13,nfilt=26,nfft=1103).T
    mfccs[c] = mel

plot_signals(signals)
#plt.show()
plot_fft(fft)
#plt.show()
plot_mfccs(mfccs)
#plt.show()

if len(os.listdir('clean')) == 0:
    for f in tqdm(df.fname):
        x,sample_rate = load('wavfiles/'+f,sr=16000)
        mask = envelope(x,sample_rate,t = 0.0005)
        wavfile.write(filename='clean/'+f,rate=sample_rate,data=x[mask])