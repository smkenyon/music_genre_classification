import os
import pandas as pd
import multiprocessing
import librosa
import scipy as sp
import zipfile as zf
import time
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf

def worker(x):
    return x**2

def feature_extraction(file, mz):
    myzip = zf.ZipFile(mz)
    del_path = myzip.extract(file)
    name = file
    try:
        x, sr = librosa.load(file, sr=None, mono=True)  # we don't want to save the waveform to file
    except:
        print("Error with librosa on: ", file)
        return 0
    # TODO: look into all of the feature extraction that we need to compute now

    # Time-Domain Features:
    
    # central moments of the amplitude calculation
    x_mean = np.mean(x)
    x_stdev = np.std(x)
    x_kurtosis = sp.stats.kurtosis(x)
    x_skew = sp.stats.skew(x)
    
    # zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(x)
    zcr_mean = np.mean(zcr)
    zcr_stdev = np.std(zcr)
    zcr_kurtosis = sp.stats.kurtosis(zcr, axis=None)
    zcr_skew = sp.stats.skew(zcr, axis=None)
    
    # rms
    rms = librosa.feature.rmse(x)
    rms_mean = np.mean(rms)
    rms_stdev = np.std(rms)
    rms_kurtosis = sp.stats.kurtosis(rms, axis=None)
    rms_skew = sp.stats.skew(rms, axis=None)
    
    # tempo (mean and variance)
    onset_env = librosa.onset.onset_strength(x, sr=sr)
    tempo = librosa.beat.tempo(x, onset_envelope=onset_env, sr=sr)[0]  # single estimated tempo
    tempo_dynamic = librosa.beat.tempo(x, onset_envelope=onset_env, sr=sr, aggregate=None)
    tempo_mean = np.mean(tempo_dynamic)
    tempo_stdev = np.std(tempo_dynamic)
    tempo_kurtosis = sp.stats.kurtosis(tempo_dynamic, axis=None)
    tempo_skew = sp.stats.kurtosis(tempo_dynamic, axis=None)
    
    # length
    piece_length = librosa.get_duration(y=x, sr=sr)
    
    # Spectral Features:
    
    # save spectrogram as its own image
    spectrogram_dir = r'D:\Analytics\spectrograms'
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))  # will reuse for remainder of spectral features
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    log_mel = librosa.amplitude_to_db(mel)  # want to save the log_mel (dB conversion)
    plt.figure(figsize=(12,4))
    ax = plt.axes()
    plt.set_cmap('hot')
    ax.set_axis_off()
    librosa.display.specshow(log_mel, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    output_name = ''
    for char in name:
        output_name += char
        if char == '/':
            output_name = ''
    output_name = output_name[:-4] + '.png'
    plt.savefig(spectrogram_dir+'\\'+output_name, bbox_inches='tight', transparent=True, pad_inches=0.0)
    
    # mel-frequency cepstral coeffs (MFCC)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    mfcc_mean = np.mean(mfcc)
    mfcc_stdev = np.std(mfcc)
    mfcc_kurtosis = sp.stats.kurtosis(mfcc, axis=None)
    mfcc_skew = sp.stats.skew(mfcc, axis=None)
    
    # chroma feature vectors
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma)  # average chroma value should correspond to key of the piece
    chroma_stdev = np.std(chroma)
    chroma_kurtosis = sp.stats.kurtosis(chroma, axis=None)
    chroma_skew = sp.stats.skew(chroma, axis=None)
    
    # spectral centroid
    S, phase = librosa.magphase(stft)  # does magphase care if the stft is alread the abs?
    spectral_centroid = librosa.feature.spectral_centroid(S=S)  # can we just pass ab
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_stdev = np.std(spectral_centroid)
    spectral_centroid_kurtosis = sp.stats.kurtosis(spectral_centroid, axis=None)
    spectral_centroid_skew = sp.stats.skew(spectral_centroid, axis=None)
    
    # spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_stdev = np.std(spectral_bandwidth)
    spectral_bandwidth_kurtosis = sp.stats.kurtosis(spectral_bandwidth, axis=None)
    spectral_bandwidth_skew = sp.stats.skew(spectral_bandwidth, axis=None)
    
    # spectral contrast
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast_mean = np.mean(contrast)
    contrast_stdev = np.std(contrast)
    contrast_kurtosis = sp.stats.kurtosis(contrast, axis=None)
    contrast_skew = sp.stats.skew(contrast, axis=None)
    
    # spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(S=S)
    rolloff_mean = np.mean(rolloff)
    rolloff_stdev = np.std(rolloff)
    rolloff_kurtosis = sp.stats.kurtosis(rolloff, axis=None)
    rolloff_skew = sp.stats.skew(rolloff, axis=None)
    
    # Additional Features that We didn't say we'd include, but might be useful:
    
    # spectral flatness
    flatness = librosa.feature.spectral_flatness(S=S) #  a high flatness means it is close to noise
    flatness_mean = np.mean(flatness)
    flatness_stdev = np.std(flatness)
    flatness_kurtosis = sp.stats.kurtosis(flatness, axis=None)
    flatness_skew = sp.stats.skew(flatness, axis=None)
    
    data_dict = {'name': [name],'x_mean': [x_mean], 'x_stdev': [x_stdev], 'x_kurtosis': [x_kurtosis], 'x_skew': [x_skew], 'zcr_mean': [zcr_mean], 'zcr_stdev': [zcr_stdev], 'zcr_kurtosis': 
                 [zcr_kurtosis], 'zcr_skew': [zcr_skew], 'rms_mean': [rms_mean], 'rms_stdev': [rms_stdev], 'rms_kurtosis': [rms_kurtosis], 'rms_skew': [rms_skew], 'tempo': 
                 [tempo], 'mean_dynamic_tempo': [tempo_mean], 'stdev_dynamic_tempo': [tempo_stdev], 'kurtosis_dynamic_tempo': [tempo_kurtosis], 'skew_dynamic_tempo': [tempo_skew],
                 'length': [piece_length], 'mfcc_mean': [mfcc_mean], 'mfcc_stdev': [mfcc_stdev], 'mfcc_kurtosis': [mfcc_kurtosis], 'mfcc_skew': [mfcc_skew], 'chroma_mean':
                 [chroma_mean], 'chroma_stdev': [chroma_stdev], 'chroma_kurtosis': [chroma_kurtosis], 'chroma_skew': [chroma_skew], 'spectral_centroid_mean': 
                 [spectral_centroid_mean], 'spectral_centroid_stdev': [spectral_centroid_stdev], 'spectral_centroid_kurtosis': [spectral_centroid_kurtosis],
                 'spectral_centroid_skew': [spectral_centroid_skew], 'spectral_bandwidth_mean': [spectral_bandwidth_mean], 'spectral_bandwidth_stdev': [spectral_bandwidth_stdev],
                 'spectral_bandwidth_kurtosis': [spectral_bandwidth_kurtosis], 'spectral_bandwidth_skew': [spectral_bandwidth_skew], 'spectral_contrast_mean': [contrast_mean],
                 'spectral_contrast_stdev': [contrast_stdev], 'spectral_contrast_kurtosis': [contrast_kurtosis], 'spectral_contrast_skew': [contrast_skew], 
                 'spectral_rolloff_mean': [rolloff_mean], 'spectral_rolloff_stdev': [rolloff_stdev], 'spectral_rolloff_kurtosis': [rolloff_kurtosis], 'spectral_rolloff_skew':
                 [rolloff_skew], 'spectral_flatness_mean': [flatness_mean], 'spectral_flatness_stdev': [flatness_stdev], 'spectral_flatness_kurtosis': [flatness_kurtosis],
                 'spectral_flatness_skew': [flatness_skew]}
    #print(data_dict)
    extracted_features = pd.DataFrame(data_dict)
    #print(extracted_features)
    extracted_features.to_csv('extracted_audio_features.csv', mode='a', header=False)
    os.remove(del_path)
    return 0
    
def create_csv():
    files = os.listdir()
    if 'extracted_audio_features.csv' not in files:
        audio_data = pd.DataFrame(columns=['names','x_mean','x_stdev','x_kurtosis','x_skew','zcr_mean','zcr_stdev','zcr_kurtosis','zcr_skew','rms_mean','rms_stdev',
                                          'rms_kurtosis','rms_skew','tempo','mean_dynamic_tempo','stdev_dynamic_tempo','kurtosis_dynamic_tempo','skew_dynamic_tempo',
                                          'length','mfcc_mean','mfcc_stdev','mfcc_kurtosis','mfcc_skew','chroma_mean','chroma_stdev','chroma_kurtosis','chroma_skew',
                                          'spectra_centroid_mean','spectral_centroid_stdev','spectral_centroid_kurtosis','spectral_centroid_skew',
                                          'spectral_bandwidth_mean','spectral_bandwidth_stdev','spectral_bandwidth_kurtosis','spectral_bandwidth_skew',
                                          'spectral_contrast_mean','spectral_contrast_stdev','spectral_contrast_kurtosis','spectral_contrast_skew','spectal_rolloff_mean',
                                          'spectral_rolloff_stdev','spectral_rolloff_kurtosis','spectral_rolloff_skew','spectral_flatness_mean',
                                          'spectral_flatness_stdev','spectral_flatness_kurtosis','spectral_flatness_skew'])
        audio_data.to_csv('extracted_audio_features.csv')

def feature_extraction_from_file(file):
    """
    Takes an input file str, extracts contents to extracted_data.csv
    """
    name = file
    try:
        x, sr = librosa.load(file, sr=None, mono=True)  # we don't want to save the waveform to file
    except:
        print("Error with librosa on: ", file)
        return 0
    # TODO: look into all of the feature extraction that we need to compute now

    # Time-Domain Features:
    
    # central moments of the amplitude calculation
    x_mean = np.mean(x)
    x_stdev = np.std(x)
    x_kurtosis = sp.stats.kurtosis(x)
    x_skew = sp.stats.skew(x)
    
    # zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(x)
    zcr_mean = np.mean(zcr)
    zcr_stdev = np.std(zcr)
    zcr_kurtosis = sp.stats.kurtosis(zcr, axis=None)
    zcr_skew = sp.stats.skew(zcr, axis=None)
    
    # rms
    rms = librosa.feature.rmse(x)
    rms_mean = np.mean(rms)
    rms_stdev = np.std(rms)
    rms_kurtosis = sp.stats.kurtosis(rms, axis=None)
    rms_skew = sp.stats.skew(rms, axis=None)
    
    # tempo (mean and variance)
    onset_env = librosa.onset.onset_strength(x, sr=sr)
    tempo = librosa.beat.tempo(x, onset_envelope=onset_env, sr=sr)[0]  # single estimated tempo
    tempo_dynamic = librosa.beat.tempo(x, onset_envelope=onset_env, sr=sr, aggregate=None)
    tempo_mean = np.mean(tempo_dynamic)
    tempo_stdev = np.std(tempo_dynamic)
    tempo_kurtosis = sp.stats.kurtosis(tempo_dynamic, axis=None)
    tempo_skew = sp.stats.kurtosis(tempo_dynamic, axis=None)
    
    # length
    piece_length = librosa.get_duration(y=x, sr=sr)
    
    # Spectral Features:
    
    # save spectrogram as its own image
    spectrogram_dir = r'D:\Analytics\spectrograms'
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))  # will reuse for remainder of spectral features
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    log_mel = librosa.amplitude_to_db(mel)  # want to save the log_mel (dB conversion)
    plt.figure(figsize=(12,4))
    ax = plt.axes()
    plt.set_cmap('hot')
    ax.set_axis_off()
    librosa.display.specshow(log_mel, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    output_name = ''
    stripped_name = name.rsplit('\\',1)[-1]
    for char in stripped_name:
        output_name += char
        if char == '/':
            output_name = ''
    output_name = output_name[:-4] + '.png'
    plt.savefig(spectrogram_dir+'\\'+output_name, bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.close()
    
    # mel-frequency cepstral coeffs (MFCC)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    mfcc_mean = np.mean(mfcc)
    mfcc_stdev = np.std(mfcc)
    mfcc_kurtosis = sp.stats.kurtosis(mfcc, axis=None)
    mfcc_skew = sp.stats.skew(mfcc, axis=None)
    
    # chroma feature vectors
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma)  # average chroma value should correspond to key of the piece
    chroma_stdev = np.std(chroma)
    chroma_kurtosis = sp.stats.kurtosis(chroma, axis=None)
    chroma_skew = sp.stats.skew(chroma, axis=None)
    
    # spectral centroid
    S, phase = librosa.magphase(stft)  # does magphase care if the stft is alread the abs?
    spectral_centroid = librosa.feature.spectral_centroid(S=S)  # can we just pass ab
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_stdev = np.std(spectral_centroid)
    spectral_centroid_kurtosis = sp.stats.kurtosis(spectral_centroid, axis=None)
    spectral_centroid_skew = sp.stats.skew(spectral_centroid, axis=None)
    
    # spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_stdev = np.std(spectral_bandwidth)
    spectral_bandwidth_kurtosis = sp.stats.kurtosis(spectral_bandwidth, axis=None)
    spectral_bandwidth_skew = sp.stats.skew(spectral_bandwidth, axis=None)
    
    # spectral contrast
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast_mean = np.mean(contrast)
    contrast_stdev = np.std(contrast)
    contrast_kurtosis = sp.stats.kurtosis(contrast, axis=None)
    contrast_skew = sp.stats.skew(contrast, axis=None)
    
    # spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(S=S)
    rolloff_mean = np.mean(rolloff)
    rolloff_stdev = np.std(rolloff)
    rolloff_kurtosis = sp.stats.kurtosis(rolloff, axis=None)
    rolloff_skew = sp.stats.skew(rolloff, axis=None)
    
    # Additional Features that We didn't say we'd include, but might be useful:
    
    # spectral flatness
    flatness = librosa.feature.spectral_flatness(S=S) #  a high flatness means it is close to noise
    flatness_mean = np.mean(flatness)
    flatness_stdev = np.std(flatness)
    flatness_kurtosis = sp.stats.kurtosis(flatness, axis=None)
    flatness_skew = sp.stats.skew(flatness, axis=None)
    
    data_dict = {'name': [name],'x_mean': [x_mean], 'x_stdev': [x_stdev], 'x_kurtosis': [x_kurtosis], 'x_skew': [x_skew], 'zcr_mean': [zcr_mean], 'zcr_stdev': [zcr_stdev], 'zcr_kurtosis': 
                 [zcr_kurtosis], 'zcr_skew': [zcr_skew], 'rms_mean': [rms_mean], 'rms_stdev': [rms_stdev], 'rms_kurtosis': [rms_kurtosis], 'rms_skew': [rms_skew], 'tempo': 
                 [tempo], 'mean_dynamic_tempo': [tempo_mean], 'stdev_dynamic_tempo': [tempo_stdev], 'kurtosis_dynamic_tempo': [tempo_kurtosis], 'skew_dynamic_tempo': [tempo_skew],
                 'length': [piece_length], 'mfcc_mean': [mfcc_mean], 'mfcc_stdev': [mfcc_stdev], 'mfcc_kurtosis': [mfcc_kurtosis], 'mfcc_skew': [mfcc_skew], 'chroma_mean':
                 [chroma_mean], 'chroma_stdev': [chroma_stdev], 'chroma_kurtosis': [chroma_kurtosis], 'chroma_skew': [chroma_skew], 'spectral_centroid_mean': 
                 [spectral_centroid_mean], 'spectral_centroid_stdev': [spectral_centroid_stdev], 'spectral_centroid_kurtosis': [spectral_centroid_kurtosis],
                 'spectral_centroid_skew': [spectral_centroid_skew], 'spectral_bandwidth_mean': [spectral_bandwidth_mean], 'spectral_bandwidth_stdev': [spectral_bandwidth_stdev],
                 'spectral_bandwidth_kurtosis': [spectral_bandwidth_kurtosis], 'spectral_bandwidth_skew': [spectral_bandwidth_skew], 'spectral_contrast_mean': [contrast_mean],
                 'spectral_contrast_stdev': [contrast_stdev], 'spectral_contrast_kurtosis': [contrast_kurtosis], 'spectral_contrast_skew': [contrast_skew], 
                 'spectral_rolloff_mean': [rolloff_mean], 'spectral_rolloff_stdev': [rolloff_stdev], 'spectral_rolloff_kurtosis': [rolloff_kurtosis], 'spectral_rolloff_skew':
                 [rolloff_skew], 'spectral_flatness_mean': [flatness_mean], 'spectral_flatness_stdev': [flatness_stdev], 'spectral_flatness_kurtosis': [flatness_kurtosis],
                 'spectral_flatness_skew': [flatness_skew]}
    #print(data_dict)
    extracted_features = pd.DataFrame(data_dict)
    #print(extracted_features)
    extracted_features.to_csv('extracted_audio_features.csv', mode='a', header=False)
    return 0

def power_to_db(S, amin=1e-16, top_db=80.0):
    """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
    Computes the scaling ``10 * log10(S / max(S))`` in a numerically
    stable way.
    Based on:
    https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    """
    def _tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    
    # Scale magnitude relative to maximum value in S. Zeros in the output 
    # correspond to positions where S == ref.
    ref = tf.reduce_max(S)

    log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec

def build_spectrogram_with_tf(file):
    try:
        x, sr = librosa.load(file, sr=None, mono=True)
    except:
        print("Error with librosa on: ", file)
        return 0
    name = file
    signals = tf.reshape(x, [1, -1])  # reshape to shape of (batch_size, samples)
    stft = tf.signal.stft(signals, frame_length=2048, frame_step=512,
                     fft_length=2048)
    mag_S = tf.abs(stft)
    num_spectrogram_bins = mag_S.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 20000, 64
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
        upper_edge_hertz)
    mel = tf.matmul(tf.square(mag_S), linear_to_mel_weight_matrix)
    log_mel = power_to_db(mel)
    plt.figure(figsize=(12,4))
    ax = plt.axes()
    plt.set_cmap('hot')
    ax.set_axis_off()
    librosa.display.specshow(log_mel[0].numpy().T, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    output_name = name.split('\\')[-1]
    output_name = output_name[:-4] + '.png'
    spectrogram_dir = r'D:\fma_large_spectrograms_tf'
    plt.savefig(spectrogram_dir+'\\'+output_name, bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.close()
    return None
    
def build_spectrogram_with_librosa(file):
    try:
        x, sr = librosa.load(file, sr=None, mono=True)
    except:
        print("Error with librosa on: ", file)
        return 0
    name = file
    spectrogram_dir = r'D:\fma_large_spectrograms_librosa'
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))  # will reuse for remainder of spectral features
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    log_mel = librosa.amplitude_to_db(mel)  # want to save the log_mel (dB conversion)
    plt.figure(figsize=(12,4))
    ax = plt.axes()
    plt.set_cmap('hot')
    ax.set_axis_off()
    librosa.display.specshow(log_mel, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    output_name = name.split("\\")[-1]
    output_name = output_name[:-4] + '.png'
    plt.savefig(spectrogram_dir+'\\'+output_name, bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.close()
    return None