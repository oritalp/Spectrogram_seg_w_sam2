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
from scipy.signal import resample
import scipy.fftpack as fft
import noisereduce as nr


def input2mel(filename, output_dir, single_frame):
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
    if single_frame:
        duration = audio.size(0) / sr
    else:
        duration = 1
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

        output_dir_img = output_dir  / "images"
        output_dir_spec = output_dir / "np_arrays"
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
    Calculate basic SDR between clean and enhanced audio signals without 
    normalization or alignment adjustments.
    
    Args:
        clean_path (str): Path to the clean audio file
        enhanced_path (str): Path to the enhanced/processed audio file
    
    Returns:
        float: SDR value in dB
    """
    # Load audio files
    clean_audio, _ = T.load(clean_path)
    enhanced_audio, _ = T.load(enhanced_path)
    
    # Convert to mono if stereo
    if clean_audio.shape[0] > 1:
        clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
    if enhanced_audio.shape[0] > 1:
        enhanced_audio = torch.mean(enhanced_audio, dim=0, keepdim=True)
    
    # Match lengths
    min_len = min(clean_audio.shape[1], enhanced_audio.shape[1])
    clean_audio = clean_audio[:, :min_len].squeeze()
    enhanced_audio = enhanced_audio[:, :min_len].squeeze()
    
    # Calculate the scaling factor that minimizes the MSE
    alpha = torch.dot(clean_audio, enhanced_audio) / (torch.dot(clean_audio, clean_audio) + 1e-8)
    
    # Calculate SDR
    scaled_reference = alpha * clean_audio
    noise = enhanced_audio - scaled_reference
    
    sdr = 10 * torch.log10(
        torch.sum(scaled_reference ** 2) / (torch.sum(noise ** 2) + 1e-8)
    )
    
    return sdr.item()

def calculate_stoi(clean_path, enhanced_path, debug=False):
    """
    Calculate STOI between clean and enhanced audio signals.
    """

    # Load audio files
    clean_audio, fs_signal = T.load(clean_path)
    enhanced_audio, _ = T.load(enhanced_path)
    
    if debug:
        print(f"Original shapes - Clean: {clean_audio.shape}, Enhanced: {enhanced_audio.shape}")
        print(f"Original sample rate: {fs_signal}")
    
    # Convert to mono if stereo
    if clean_audio.shape[0] > 1:
        clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
    if enhanced_audio.shape[0] > 1:
        enhanced_audio = torch.mean(enhanced_audio, dim=0, keepdim=True)
    
    # Convert to numpy and squeeze
    clean = clean_audio.squeeze().numpy()
    enhanced = enhanced_audio.squeeze().numpy()
    
    # Remove DC offset
    clean = clean - np.mean(clean)
    enhanced = enhanced - np.mean(enhanced)
    
    # Normalize signals to have unit energy
    clean = clean / np.sqrt(np.sum(clean**2))
    enhanced = enhanced / np.sqrt(np.sum(enhanced**2))
    
    if debug:
        print(f"Signal energies after normalization - Clean: {np.sum(clean**2):.3f}, "
              f"Enhanced: {np.sum(enhanced**2):.3f}")
    
    # Match lengths
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]
    
    # Parameters
    fs = 10000  # Target sampling rate
    N = 256     # DFT length
    K = 384     # Segment length (changed from 512 to match original paper)
    J = 15      # Number of 1/3 octave bands
    Beta = -15  # Lower SDR bound
    dyn_range = 40  # Speech dynamic range in dB
    
    # Resample to 10 kHz if needed
    if fs_signal != fs:
        clean = resample(clean, int(len(clean) * fs / fs_signal))
        enhanced = resample(enhanced, int(len(enhanced) * fs / fs_signal))
    
    # Short-time segmentation parameters
    N_frame = 256    # Frame length
    K_frame = 384    # Total segment length
    H = N_frame//2   # Hop length
    
    # Number of frames
    num_frames = (len(clean) - K_frame) // H + 1
    
    # Initialize 1/3 octave band parameters
    cf = 150 * 2**(np.arange(J)/3)  # Center frequencies
    erb = 24.7 * (4.37 * cf / 1000 + 1)  # Equivalent rectangular bandwidth
    
    # Set up analysis and synthesis windows
    win = np.hanning(N_frame)
    win = win / np.sqrt(np.sum(win**2))
    
    # Initialize matrices for TF decomposition
    X = np.zeros((J, num_frames))
    Y = np.zeros((J, num_frames))

    # Short-time Fourier analysis
    for frame_idx in range(num_frames):
        start_idx = frame_idx * H
        stop_idx = start_idx + N_frame
        
        x_frame = clean[start_idx:stop_idx] * win
        y_frame = enhanced[start_idx:stop_idx] * win
        
        x_stft = fft.fft(x_frame, N)[:N//2 + 1]
        y_stft = fft.fft(y_frame, N)[:N//2 + 1]
        
        # Power spectra
        x_power = np.abs(x_stft)**2
        y_power = np.abs(y_stft)**2
        
        # Apply ERB-like weighting
        for j in range(J):
            cf_j = cf[j]
            erb_j = erb[j]
            
            # Frequency weighting
            w = np.exp(-0.5 * ((fs * np.arange(N//2 + 1) / N - cf_j) / (erb_j/2.0))**2)
            w = w / np.sum(w)
            
            # Apply weighting and sum
            X[j, frame_idx] = np.sqrt(np.sum(x_power * w))
            Y[j, frame_idx] = np.sqrt(np.sum(y_power * w))
    
    if debug:
        print(f"TF representation shapes - X: {X.shape}, Y: {Y.shape}")
    
    # Normalize representations
    for j in range(J):
        X[j, :] = (X[j, :] - np.mean(X[j, :])) / (np.std(X[j, :]) + 1e-8)
        Y[j, :] = (Y[j, :] - np.mean(Y[j, :])) / (np.std(Y[j, :]) + 1e-8)
    
    # Apply intensity clipping
    c = 10**(Beta/20)
    X = np.maximum(X, c * np.max(X))
    Y = np.maximum(Y, c * np.max(Y))

    # Compute correlation coefficients for each band
    d = np.zeros(J)
    for j in range(J):
        x = X[j, :]
        y = Y[j, :]
        
        # Compute correlation
        d[j] = np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2) + 1e-8)
    
    if debug:
        print(f"Band-wise correlations: {d}")
        print(f"Mean correlation: {np.mean(d):.3f}")
    
    # Ensure final score is between 0 and 1
    final_score = float(np.mean(d))
    final_score = np.clip(final_score, 0.0, 1.0)
    
    return final_score

def print_quality_metrics(clean_path, noisy_path, *enhanced_paths, debug=False):
    """
    Print SDR and STOI comparison between noisy audio and variable number of enhanced audio files.
    
    Args:
        clean_path (str): Path to the clean reference audio
        noisy_path (str): Path to the noisy audio
        *enhanced_paths (str): Variable number of paths to enhanced audio files
        debug (bool): Boolean flag for additional debug information
    """
    import os
    
    def get_method_name(path):
        """Extract method name from file path."""
        basename = os.path.basename(path)  # Get filename from path
        name = os.path.splitext(basename)[0]  # Remove extension
        # Replace underscores/hyphens with spaces and capitalize
        return name.replace('_', ' ').replace('-', ' ').title()
    
    # Calculate metrics for noisy audio
    print("Calculating metrics for noisy audio...")
    noisy_sdr = calculate_sdr(clean_path, noisy_path)
    noisy_stoi = calculate_stoi(clean_path, noisy_path, debug=debug)
    
    # Calculate metrics for all enhanced audio files
    enhanced_metrics = []
    for path in enhanced_paths:
        method_name = get_method_name(path)
        print(f"Calculating metrics for {method_name}...")
        
        sdr = calculate_sdr(clean_path, path)
        stoi = calculate_stoi(clean_path, path, debug=debug)
        sdr_improvement = sdr - noisy_sdr
        stoi_improvement = stoi - noisy_stoi
        
        enhanced_metrics.append({
            'name': method_name,
            'sdr': sdr,
            'stoi': stoi,
            'sdr_improvement': sdr_improvement,
            'stoi_improvement': stoi_improvement
        })
    
    # Calculate maximum name length for proper alignment
    max_name_length = max(
        len("Noisy"),
        max(len(m['name']) for m in enhanced_metrics)
    ) + 2  # Add minimal padding
    
    # Define column widths
    sdr_width = 8
    stoi_width = 10
    improvement_width = 12
    
    # Print the table
    print("\nAudio Quality Metrics:")
    print("-" * (max_name_length + sdr_width + stoi_width + 2 * improvement_width + 10))
    
    # Headers
    method_col = f"{{:<{max_name_length}}}"
    main_header = (
        f"{method_col}     {'SDR (dB)':<{sdr_width}} {'STOI':<{stoi_width}}     "
        f"{'Improvement':>{2 * improvement_width}}"
    ).format("Method")
    print(main_header)
    
    sub_header = (
        f"{method_col}     {'':>{sdr_width}} {'':>{stoi_width}}     "
        f"{'SDR (dB)':>{improvement_width}} {'STOI':>{improvement_width}}"
    ).format("")
    print(sub_header)
    
    print("-" * (max_name_length + sdr_width + stoi_width + 2 * improvement_width + 10))
    
    # Print noisy baseline
    print(
        f"{method_col}     {noisy_sdr:>6.2f} {'dB':<2} {noisy_stoi:>8.5f}".format("Noisy")
    )
    
    # Print enhanced metrics
    for metrics in enhanced_metrics:
        print(
            f"{method_col}     {metrics['sdr']:>6.2f} {'dB':<2} {metrics['stoi']:>8.5f}     "
            f"{metrics['sdr_improvement']:>10.2f} {metrics['stoi_improvement']:>12.5f}".format(metrics['name'])
        )
    
    print("-" * (max_name_length + sdr_width + stoi_width + 2 * improvement_width + 10))

def reduce_noise(input_file, output_file, decrease_factor=0.75):
    # Load audio using torchaudio
    waveform, sample_rate = T.load(input_file)
    
    # Convert to mono if stereo (torchaudio loads as [channels, samples])
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)
    
    # Convert to numpy array (noisereduce expects numpy)
    audio_np = waveform.numpy()
    
    # Reduce noise - non-stationary mode
    reduced_noise = nr.reduce_noise(
        y=audio_np,
        sr=sample_rate,
        stationary=False,  # For varying background noise
        prop_decrease=decrease_factor)
    
    # Convert back to torch tensor for saving
    reduced_noise_tensor = torch.FloatTensor(reduced_noise)
    
    # Save using torchaudio
    T.save(output_file, reduced_noise_tensor, sample_rate)