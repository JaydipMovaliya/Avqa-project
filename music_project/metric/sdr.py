import numpy as np
import mir_eval

def compute_sdr(reference_audio, estimated_audio, sr=16000):
    """
    Computes SDR using mir_eval library.

    Args:
        reference_audio (np.ndarray): Ground truth audio, shape (T,)
        estimated_audio (np.ndarray): Predicted separated audio, shape (T,)
        sr (int): Sample rate (default 16k)

    Returns:
        sdr (float): Signal-to-Distortion Ratio (dB)
    """
    # Ensure inputs are 2D arrays with shape (1, T)
    reference_audio = np.asarray(reference_audio).reshape(1, -1)
    estimated_audio = np.asarray(estimated_audio).reshape(1, -1)

    # Align lengths
    min_len = min(reference_audio.shape[-1], estimated_audio.shape[-1])
    reference_audio = reference_audio[:, :min_len]
    estimated_audio = estimated_audio[:, :min_len]

    # mir_eval requires both reference and estimate
    sdr, _, _, _ = mir_eval.separation.bss_eval_sources(reference_audio, estimated_audio)
    return sdr[0]  # return scalar
