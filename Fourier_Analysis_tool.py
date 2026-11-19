"""
Fourier analysis for signal processing.

Use it from the command line or import it as a module.

Requires: numpy, matplotlib   (pip install numpy matplotlib)

Command-line examples
---------------------
  python fourier_tool.py --demo
  python fourier_tool.py --tones 50:1,120:0.5,333:0.2 --noise 0.1 --fs 1000
  python fourier_tool.py recording.wav --window blackman --spectrogram
  python fourier_tool.py data.csv --fs 2000 --bandpass 40 150 --save result.png
  python fourier_tool.py data.csv --fs 2000 --export spectrum.csv

Module example
--------------
  from fourier_tool import generate, amplitude_spectrum, find_peaks
  x = generate(fs=1000, duration=1, tones=[(50, 1), (120, 0.5)], noise=0.1)
  freqs, amp = amplitude_spectrum(x, fs=1000, window="hann")
  print(find_peaks(freqs, amp))
"""
import argparse
import sys
import wave

import numpy as np

# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------

def get_window(name, n):
    """Return a periodic window of length n (suited to spectral analysis)."""
    name = name.lower()
    k = np.arange(n)
    x = 2 * np.pi * k / n
    if name in ("rect", "rectangular", "boxcar", "none"):
        return np.ones(n)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(x)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(x)
    if name == "blackman":
        return 0.42 - 0.5 * np.cos(x) + 0.08 * np.cos(2 * x)
    if name in ("flattop", "flat-top"):
        return (0.21557895 - 0.41663158 * np.cos(x) + 0.277263158 * np.cos(2 * x)
                - 0.083578947 * np.cos(3 * x) + 0.006947368 * np.cos(4 * x))
    raise ValueError(f"Unknown window: {name!r}")


WINDOW_NAMES = ["rect", "hann", "hamming", "blackman", "flattop"]

# --------------------------------------------------------------------------
# Signal sources
# --------------------------------------------------------------------------

def generate(fs, duration, tones=((50, 1.0),), noise=0.0, seed=0):
    """Sum of sines (frequency Hz, amplitude) plus Gaussian noise of std `noise`."""
    t = np.arange(int(round(fs * duration))) / fs
    x = np.zeros_like(t)
    for f, a in tones:
        x += a * np.sin(2 * np.pi * f * t)
    if noise > 0:
        x += noise * np.random.default_rng(seed).standard_normal(t.size)
    return x


def load_signal(path, fs=None):
    """Load a .wav (first channel) or a text/CSV file (last numeric column).

    Returns (samples, sample_rate). For text files, `fs` must be supplied.
    """
    if path.lower().endswith(".wav"):
        with wave.open(path, "rb") as w:
            rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
            raw = w.readframes(w.getnframes())
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
        if dtype is None:
            raise ValueError("Only 8, 16 and 32-bit PCM WAV files are supported.")
        data = np.frombuffer(raw, dtype=dtype).reshape(-1, ch)[:, 0].astype(float)
        if width == 1:
            data -= 128
        return data / float(2 ** (8 * width - 1)), (fs or rate)

    if fs is None:
        raise ValueError("--fs is required for CSV/text input.")
    values = []
    with open(path) as fh:
        for line in fh:
            parts = [p for p in line.replace(";", ",").replace("\t", ",").replace(" ", ",").split(",") if p]
            try:
                values.append(float(parts[-1]))
            except (ValueError, IndexError):
                continue  # skip headers and blank lines
    if len(values) < 8:
        raise ValueError("Found fewer than 8 numeric samples in the file.")
    return np.array(values), fs

# --------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------

def amplitude_spectrum(x, fs, window="hann", nfft=None, detrend=True):
    """Single-sided amplitude spectrum.

    Scaled so a sine of amplitude A shows a peak of height A (for any window,
    when the tone sits near a bin center). Returns (freqs_hz, amplitude).
    """
    x = np.asarray(x, dtype=float)
    if detrend:
        x = x - x.mean()
    n = len(x)
    w = get_window(window, n)
    nfft = nfft or n
    spec = np.fft.rfft(x * w, nfft)
    amp = np.abs(spec) * 2.0 / w.sum()
    amp[0] /= 2.0
    if nfft % 2 == 0:
        amp[-1] /= 2.0
    return np.fft.rfftfreq(nfft, 1.0 / fs), amp


def to_db(amp, floor=1e-12):
    return 20 * np.log10(np.maximum(amp, floor))


def find_peaks(freqs, amp, n=6, rel_threshold=0.08):
    """Strongest local maxima, refined by parabolic interpolation in dB.

    Returns a list of (frequency_hz, amplitude, level_db), strongest first.
    """
    db = to_db(amp)
    thr = rel_threshold * amp[1:].max()
    df = freqs[1] - freqs[0]
    peaks = []
    for k in range(2, len(amp) - 1):
        if amp[k] > thr and amp[k] > amp[k - 1] and amp[k] >= amp[k + 1]:
            a, b, c = db[k - 1], db[k], db[k + 1]
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if den else 0.0
            level = b - 0.25 * (a - c) * d
            peaks.append((freqs[k] + d * df, 10 ** (level / 20), level))
    return sorted(peaks, key=lambda p: -p[1])[:n]


def frequency_filter(x, fs, low=None, high=None):
    """Zero-phase FFT filter. Keeps frequencies in [low, high] Hz.

    low only  -> high-pass;  high only -> low-pass;  both -> band-pass.
    Note: sharp cutoffs cause ringing; fine for analysis, not for real-time use.
    """
    x = np.asarray(x, dtype=float)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    mask = np.ones_like(f, dtype=bool)
    if low is not None:
        mask &= f >= low
    if high is not None:
        mask &= f <= high
    return np.fft.irfft(spec * mask, len(x))


def spectrogram(x, fs, nperseg=256, overlap=0.5, window="hann"):
    """Short-time Fourier transform. Returns (freqs, times, amplitude[freq, time])."""
    x = np.asarray(x, dtype=float)
    nperseg = min(nperseg, len(x))
    step = max(1, int(nperseg * (1 - overlap)))
    w = get_window(window, nperseg)
    starts = range(0, len(x) - nperseg + 1, step)
    cols = [np.abs(np.fft.rfft(x[s:s + nperseg] * w)) * 2 / w.sum() for s in starts]
    times = np.array([s + nperseg / 2 for s in starts]) / fs
    return np.fft.rfftfreq(nperseg, 1.0 / fs), times, np.array(cols).T

# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_analysis(x, fs, window="hann", filtered=None, show_spectrogram=False,
                  db=True, save=None, title=None):
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 3 if show_spectrogram else 2
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3.2 * rows), constrained_layout=True)
    t = np.arange(len(x)) / fs * 1000

    ax = axes[0]
    ax.plot(t, x, lw=1, color="#8aa0ad", label="original")
    if filtered is not None:
        ax.plot(t, filtered, lw=1.3, color="#0e6f8c", label="filtered")
    ax.set(xlabel="Time (ms)", ylabel="Amplitude", title=title or "Time domain")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    freqs, amp = amplitude_spectrum(x, fs, window)
    ax.plot(freqs, to_db(amp) if db else amp, lw=1.2, color="#b0237a", label="original")
    if filtered is not None:
        _, amp_f = amplitude_spectrum(filtered, fs, window)
        ax.plot(freqs, to_db(amp_f) if db else amp_f, lw=1.2, color="#0e6f8c", label="filtered")
        ax.legend(loc="upper right")
    for f, a, lv in find_peaks(freqs, amp):
        ax.annotate(f"{f:.1f} Hz", (f, lv if db else a), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    if db:
        top = to_db(amp).max()
        ax.set_ylim(top - 100, top + 10)
    ax.set(xlabel="Frequency (Hz)", ylabel="Amplitude (dB)" if db else "Amplitude",
           title=f"Amplitude spectrum ({window} window)")
    ax.grid(alpha=0.3)

    if show_spectrogram:
        ax = axes[2]
        f, tt, S = spectrogram(x, fs, window=window)
        im = ax.pcolormesh(tt, f, to_db(S), shading="auto", cmap="viridis")
        ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", title="Spectrogram")
        fig.colorbar(im, ax=ax, label="dB")

    if save:
        fig.savefig(save, dpi=150)
        print(f"Saved plot to {save}")
    else:
        plt.show()

# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def parse_tones(text):
    tones = []
    for item in text.split(","):
        f, _, a = item.partition(":")
        tones.append((float(f), float(a) if a else 1.0))
    return tones


def main(argv=None):
    p = argparse.ArgumentParser(description="Fourier analysis of a signal.")
    p.add_argument("input", nargs="?", help=".wav or .csv/.txt file (omit to generate a signal)")
    p.add_argument("--fs", type=float, help="sample rate in Hz (required for CSV; default 1000 when generating)")
    p.add_argument("--demo", action="store_true", help="analyze a built-in 50/120/333 Hz test signal")
    p.add_argument("--tones", help="generated tones as freq:amp,freq:amp (e.g. 50:1,120:0.5)")
    p.add_argument("--duration", type=float, default=1.0, help="generated signal length in seconds")
    p.add_argument("--noise", type=float, default=0.0, help="generated noise standard deviation")
    p.add_argument("--window", default="hann", choices=WINDOW_NAMES)
    p.add_argument("--linear", action="store_true", help="linear amplitude axis instead of dB")
    p.add_argument("--lowpass", type=float, metavar="HZ")
    p.add_argument("--highpass", type=float, metavar="HZ")
    p.add_argument("--bandpass", type=float, nargs=2, metavar=("LOW", "HIGH"))
    p.add_argument("--spectrogram", action="store_true")
    p.add_argument("--export", metavar="CSV", help="write the spectrum to a CSV file")
    p.add_argument("--save", metavar="PNG", help="save the figure instead of opening a window")
    p.add_argument("--no-plot", action="store_true", help="print the peak table only")
    a = p.parse_args(argv)

    if a.input:
        x, fs = load_signal(a.input, a.fs)
        label = a.input
    else:
        fs = a.fs or 1000.0
        tones = parse_tones(a.tones) if a.tones else [(50, 1.0), (120, 0.5), (333, 0.2)]
        noise = a.noise if (a.tones or a.noise) else 0.1
        x = generate(fs, a.duration, tones, noise)
        label = "generated signal"

    lo = hi = None
    if a.bandpass:
        lo, hi = a.bandpass
    if a.highpass is not None:
        lo = a.highpass
    if a.lowpass is not None:
        hi = a.lowpass
    filtered = frequency_filter(x, fs, lo, hi) if (lo is not None or hi is not None) else None

    freqs, amp = amplitude_spectrum(x, fs, a.window)
    print(f"{label}: {len(x)} samples at {fs:g} Hz ({len(x)/fs:.3f} s), "
          f"bin spacing {freqs[1]-freqs[0]:.4g} Hz")
    print(f"{'Frequency (Hz)':>16} {'Amplitude':>10} {'Level (dB)':>11}")
    for f, amp_v, lv in find_peaks(freqs, amp):
        print(f"{f:16.3f} {amp_v:10.4f} {lv:11.2f}")

    if a.export:
        np.savetxt(a.export, np.column_stack([freqs, amp, to_db(amp)]), delimiter=",",
                   header="frequency_hz,amplitude,level_db", comments="")
        print(f"Wrote spectrum to {a.export}")

    if not a.no_plot:
        plot_analysis(x, fs, a.window, filtered, a.spectrogram, not a.linear, a.save, title=label)
    return 0


if __name__ == "__main__":
    sys.exit(main())