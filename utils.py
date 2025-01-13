import os
import sys
import numpy as np
import shutil
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torchaudio as T
import torchaudio.transforms as TT
from diffwave.inference import predict as diffwave_predict
from diffwave.params import params
import librosa.display
from sam2.build_sam import build_sam2_video_predictor
from pathlib import Path
from clicker import collect_clicks
import pickle
from mir_eval.separation import bss_eval_sources
from scipy.signal import resample
import scipy.fftpack as fft


def input2mel(filename, output_dir):
    # Load the original audio file
    audio, sr = T.load(filename)  # Load audio and sample rate using torchaudio

    # Check if resampling is needed
    if sr != params.sample_rate:
        resampler = TT.Resample(orig_freq=sr, new_freq=params.sample_rate)
        audio = resampler(audio)
        sr = params.sample_rate

    # Check if reformating is needed
    if audio.type() != torch.int16:
        # Convert the audio to 16-bit PCM format
        audio = (audio * 32767).to(torch.int16)

    audio = torch.clamp(audio[0] / 32767.5, -1.0, 1.0)

    window_size = params.hop_samples * 4     
    overlap = params.hop_samples        
    nfft = params.n_fft  

    mel_args = {
        'sample_rate': sr,
        'win_length': window_size, 
        'hop_length': overlap,   
        'n_fft': nfft,
        'f_min': 20.0,
        'f_max': sr / 2.0,
        'n_mels': params.n_mels,
        'power': 1.0,
        'normalized': True,
    }
    mel_spec_transform = TT.MelSpectrogram(**mel_args)

    duration = audio.size(0) / sr
    # Calculate the total number of X-second chunks - for better interpretability
    chunk_duration = duration  # Duration in seconds
    chunk_samples = chunk_duration * sr
    total_chunks = int(np.ceil(len(audio) / chunk_samples))

    for chunk_idx in range(total_chunks):
        # Extract the current segment
         # Extract the current chunk
        start_sample = int(chunk_idx * chunk_samples)
        end_sample = int(min((chunk_idx + 1) * chunk_samples, len(audio)))
        chunk = audio[start_sample:end_sample]

        # Pad the last chunk if it's less than X seconds
        if len(chunk) < chunk_samples:
            chunk = torch.cat((chunk, torch.zeros(chunk_samples - len(chunk))))


        # Compute the spectrogram
        with torch.no_grad():
            spectrogram = mel_spec_transform(chunk)
            spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
            spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)


        # Create figure with the correct aspect ratio
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        ax.axis("off")

        librosa.display.specshow(
            spectrogram.numpy(),
            sr=sr,
            hop_length=overlap,
            x_axis="time",
            y_axis="mel",
            cmap="magma",  # Can try also 'jet'
        )

        output_dir_img = output_dir + "_images"
        output_dir_spec = output_dir + "_np_array"
        os.makedirs(output_dir_spec, exist_ok=True)
        os.makedirs(output_dir_img, exist_ok=True)

        # Save the spectrogram image
        output_path_img = os.path.join(output_dir_img, f'{chunk_idx:04d}.jpg')
        fig.savefig(output_path_img, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Save the original spectrogram data
        output_path_spec = os.path.join(output_dir_spec, f'{chunk_idx:04d}.npy')
        np.save(output_path_spec, spectrogram.numpy()) # The spectrogram is a numpy array
    return sr, overlap, output_dir_img


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def save_lists(list1, list2, filename='lists.pkl'):
    """
    Save Python lists using pickle
    
    Parameters:
        list1, list2: Python lists (can contain any type)
        filename: output filename (should end with .pkl)
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump((list1, list2), f)
        print(f"Lists successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving lists: {e}")

def load_lists(filename='lists.pkl'):
    """Load lists from pickle file"""
    try:
        with open(filename, 'rb') as f:
            list1, list2 = pickle.load(f)
        return list1, list2
    except Exception as e:
        print(f"Error loading lists: {e}")
        return None, None
    

def pixel_to_spectrogram(pixel_x, pixel_y, img_height, img_width, time_bins, freq_bins):
    # Flip y-axis (since images are top-left origin)
    spectrogram_freq_bin = freq_bins - int(pixel_y * freq_bins / img_height) - 1
    spectrogram_time_bin = int(pixel_x * time_bins / img_width)

    # Ensure indices are within bounds
    spectrogram_freq_bin = np.clip(spectrogram_freq_bin, 0, freq_bins - 1)
    spectrogram_time_bin = np.clip(spectrogram_time_bin, 0, time_bins - 1)

    return spectrogram_time_bin, spectrogram_freq_bin

def matrix_to_spectrogram(pixel_x, pixel_y, img_height, img_width, time_bins, freq_bins):
    """
    Map matrices of pixel coordinates (pixel_x, pixel_y) to spectrogram indices.

    Args:
        pixel_x (np.ndarray): Array of x-coordinates in the image.
        pixel_y (np.ndarray): Array of y-coordinates in the image.
        img_height (int): Height of the image.
        img_width (int): Width of the image.
        time_bins (int): Number of time bins in the spectrogram.
        freq_bins (int): Number of frequency bins in the spectrogram.

    Returns:
        np.ndarray: Array of time indices for the spectrogram.
        np.ndarray: Array of frequency indices for the spectrogram.
    """
    # Flip y-axis (since images are top-left origin)
    spectrogram_freq_bin = freq_bins - ((pixel_y * freq_bins / img_height).astype(int)) - 1
    spectrogram_time_bin = (pixel_x * time_bins / img_width).astype(int)

    # Ensure indices are within bounds
    spectrogram_freq_bin = np.clip(spectrogram_freq_bin, 0, freq_bins - 1)
    spectrogram_time_bin = np.clip(spectrogram_time_bin, 0, time_bins - 1)

    return spectrogram_time_bin, spectrogram_freq_bin


def calculate_sdr(clean_path, enhanced_path):
    """
    Calculate Scale-Invariant SDR between clean and enhanced audio signals.
    
    Args:
        clean_path (str): Path to the clean audio file
        enhanced_path (str): Path to the enhanced/processed audio file
    
    Returns:
        float: SI-SDR value in dB
    """
    # Load audio files
    clean_audio, _ = T.load(clean_path)
    enhanced_audio, _ = T.load(enhanced_path)
    
    # Convert to mono if stereo
    if clean_audio.shape[0] > 1:
        clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
    if enhanced_audio.shape[0] > 1:
        enhanced_audio = torch.mean(enhanced_audio, dim=0, keepdim=True)
    
    # Match lengths (take shorter length)
    min_len = min(clean_audio.shape[1], enhanced_audio.shape[1])
    clean_audio = clean_audio[:, :min_len]
    enhanced_audio = enhanced_audio[:, :min_len]
    
    # Remove mean of both signals
    clean_audio = clean_audio.squeeze() - torch.mean(clean_audio)
    enhanced_audio = enhanced_audio.squeeze() - torch.mean(enhanced_audio)
    
    # Normalize both signals
    clean_audio = clean_audio / torch.sqrt(torch.sum(clean_audio ** 2))
    enhanced_audio = enhanced_audio / torch.sqrt(torch.sum(enhanced_audio ** 2))
    
    # Calculate the scaling factor
    alpha = torch.dot(clean_audio, enhanced_audio)
    
    # Calculate target and noise
    target = alpha * clean_audio
    noise = enhanced_audio - target
    
    # Calculate SI-SDR
    sdr = 10 * torch.log10(
        torch.sum(target ** 2) / (torch.sum(noise ** 2) + 1e-8)
    )
    
    return sdr.item()


def calculate_stoi(clean_path, enhanced_path):
    """
    Calculate STOI between clean and enhanced audio signals.
    
    Args:
        clean_path (str): Path to the clean audio file
        enhanced_path (str): Path to the enhanced/processed audio file
    
    Returns:
        float: STOI score between 0 and 1
    """
    import numpy as np
    import torchaudio as T
    from scipy.signal import resample
    import scipy.fftpack as fft

    # Load audio files
    clean_audio, fs_signal = T.load(clean_path)
    enhanced_audio, _ = T.load(enhanced_path)
    
    # Convert to mono if stereo
    if clean_audio.shape[0] > 1:
        clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
    if enhanced_audio.shape[0] > 1:
        enhanced_audio = torch.mean(enhanced_audio, dim=0, keepdim=True)
    
    # Convert to numpy and squeeze
    clean = clean_audio.squeeze().numpy()
    enhanced = enhanced_audio.squeeze().numpy()
    
    # Match lengths (take shorter length)
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]
    
    # Parameters
    fs = 10000  # Target sampling rate
    N = 256    # DFT length
    K = 512    # Segment length
    J = 15     # Number of 1/3 octave bands
    
    # Resample to 10 kHz if needed
    if fs_signal != fs:
        clean = resample(clean, int(len(clean) * fs / fs_signal))
        enhanced = resample(enhanced, int(len(enhanced) * fs / fs_signal))
    
    # Initialize 1/3 octave band parameters
    cf = 150 * 2 ** (np.arange(J) / 3)
    k = np.arange(N/2 + 1)
    f = k * fs / N
    
    # Set up OBM (Octave Band Matrix)
    obm = np.zeros((J, len(f)))
    for i in range(J):
        f_low = cf[i] * 2 ** (-1/6)
        f_high = cf[i] * 2 ** (1/6)
        obm[i, np.logical_and(f >= f_low, f <= f_high)] = 1
    
    # Normalize OBM
    obm = obm / np.sum(obm, axis=1, keepdims=True)
    
    # Hanning window
    win = np.hanning(K)
    
    # Process frames
    nframes = max(1, len(clean) - K + 1)
    X = np.zeros((J, nframes))
    Y = np.zeros((J, nframes))
    
    for m in range(nframes):
        x_seg = clean[m:m+K] * win
        y_seg = enhanced[m:m+K] * win
        
        X_frame = np.abs(fft.fft(x_seg, N)[:N//2 + 1]) ** 2
        Y_frame = np.abs(fft.fft(y_seg, N)[:N//2 + 1]) ** 2
        
        X[:, m] = np.sqrt(np.dot(obm, X_frame))
        Y[:, m] = np.sqrt(np.dot(obm, Y_frame))
    
    # Apply threshold
    c = 5.62341325  # 10^(-Beta/20), with Beta = -15 dB
    X = np.maximum(X, c * np.max(X))
    Y = np.maximum(Y, c * np.max(Y))
    
    # Compute correlation coefficients
    d = np.zeros(J)
    for i in range(J):
        x = X[i, :]
        y = Y[i, :]
        
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)
        
        d[i] = np.sum(x * y) / len(x)
    
    return float(np.mean(d))

def print_quality_metrics(clean_path, noisy_path, enhanced_path):
    """
    Print SDR and STOI comparison between noisy and enhanced audio.
    
    Args:
        clean_path (str): Path to the clean audio file
        noisy_path (str): Path to the noisy audio file
        enhanced_path (str): Path to the enhanced/processed audio file
    """
    # Calculate metrics for noisy audio
    print("Calculating metrics for noisy audio...")
    noisy_sdr = calculate_sdr(clean_path, noisy_path)
    noisy_stoi = calculate_stoi(clean_path, noisy_path)
    
    # Calculate metrics for enhanced audio
    print("Calculating metrics for enhanced audio...")
    enhanced_sdr = calculate_sdr(clean_path, enhanced_path)
    enhanced_stoi = calculate_stoi(clean_path, enhanced_path)
    
    # Calculate improvements
    sdr_improvement = enhanced_sdr - noisy_sdr
    stoi_improvement = enhanced_stoi - noisy_stoi
    
    # Print results
    print("\nAudio Quality Metrics:")
    print("-" * 50)
    print(f"{'Signal':<15} {'SDR (dB)':<12} {'STOI':<10}")
    print("-" * 50)
    print(f"{'Noisy':<15} {noisy_sdr:>7.2f}{'dB':>5} {noisy_stoi:>10.3f}")
    print(f"{'Enhanced':<15} {enhanced_sdr:>7.2f}{'dB':>5} {enhanced_stoi:>10.3f}")
    print("-" * 50)
    print(f"{'Improvement':<15} {sdr_improvement:>+7.2f}{'dB':>5} {stoi_improvement:>+10.3f}")

# def print_sdr_comparison(clean_path, noisy_path, enhanced_path):
#     """
#     Print SDR comparison between noisy and enhanced audio.
    
#     Args:
#         clean_path (str): Path to the clean audio file
#         noisy_path (str): Path to the noisy audio file
#         enhanced_path (str): Path to the enhanced/processed audio file
#     """
#     # Calculate SDR for noisy audio
#     print("Calculating SDR for noisy audio...")
#     noisy_sdr = calculate_sdr(clean_path, noisy_path)
    
#     # Calculate SDR for enhanced audio
#     print("Calculating SDR for enhanced audio...")
#     enhanced_sdr = calculate_sdr(clean_path, enhanced_path)
    
#     # Calculate improvement
#     sdr_improvement = enhanced_sdr - noisy_sdr
    
#     # Print results
#     print("\nSDR Comparison:")
#     print("-" * 40)
#     print(f"{'Signal':<15} {'SDR (dB)':<10}")
#     print("-" * 40)
#     print(f"{'Noisy':<15} {noisy_sdr:.2f}")
#     print(f"{'Enhanced':<15} {enhanced_sdr:.2f}")
#     print("-" * 40)
#     print(f"{'Improvement':<15} {sdr_improvement:+.2f}")
