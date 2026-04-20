"""
OpenBCI Ganglion - BrainFlow streaming script with real-time plot
-----------------------------------------------------------------
Requirements:
    pip install brainflow numpy matplotlib

Usage:
    python ganglion_stream.py                        # USB dongle, COM3
    python ganglion_stream.py --port COM5            # USB dongle, COM5
    python ganglion_stream.py --native               # native BLE
    python ganglion_stream.py --native --output data.csv
"""

import argparse
import time
import sys
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_SERIAL_PORT = "COM3"
GANGLION_MAC        = ""
USE_NATIVE_BLE      = False
STREAM_DURATION_SEC = 0          # 0 = stream until window is closed
WINDOW_SECONDS      = 5          # seconds of data shown in the plot
NOTCH_FREQ          = 50.0       # power line frequency: 50 Hz (EU) or 60 Hz (US)
BANDPASS_LOW        = 1.0        # Hz — removes DC drift and ECG baseline
BANDPASS_HIGH       = 50.0       # Hz — removes high-freq noise
# ─────────────────────────────────────────────────────────────────────────────

CH_COLORS = ["#00bfff", "#ff6b6b", "#6bff6b", "#ffd700"]
CH_LABELS = ["CH1", "CH2", "CH3", "CH4"]


def build_params(port, mac, native_ble):
    params = BrainFlowInputParams()
    if native_ble:
        params.mac_address = mac
    else:
        params.serial_port = port
    return params


def get_board_id(native_ble):
    return BoardIds.GANGLION_NATIVE_BOARD if native_ble else BoardIds.GANGLION_BOARD


# ── Streaming thread ─────────────────────────────────────────────────────────

class GanglionStreamer(threading.Thread):
    def __init__(self, board, eeg_channels, sample_rate, window_size, duration, output_file):
        super().__init__(daemon=True)
        self.board        = board
        self.eeg_channels = eeg_channels
        self.sample_rate  = sample_rate
        self.window_size  = window_size
        self.duration     = duration
        self.output_file  = output_file

        # Rolling buffers — one deque per channel
        self.buffers = [deque(np.zeros(window_size), maxlen=window_size)
                        for _ in eeg_channels]
        self.all_data   = []        # full recording for CSV
        self.running    = True
        self.start_time = None
        self.elapsed    = 0.0

    def stop(self):
        self.running = False

    def run(self):
        self.start_time = time.time()
        try:
            while self.running:
                time.sleep(0.04)    # poll at ~25 Hz

                data = self.board.get_board_data()
                if data.shape[1] == 0:
                    continue

                self.all_data.append(data)

                for i, ch in enumerate(self.eeg_channels):
                    chunk = data[ch].copy()
                    # Bandpass 1–50 Hz: removes DC drift, ECG baseline, and high-freq noise
                    DataFilter.perform_bandpass(chunk, self.sample_rate,
                                               BANDPASS_LOW, BANDPASS_HIGH,
                                               4, 1, 0.0)
                    # Notch at 50/60 Hz: removes power line interference
                    DataFilter.perform_bandstop(chunk, self.sample_rate,
                                               NOTCH_FREQ - 2.0, NOTCH_FREQ + 2.0,
                                               4, 1, 0.0)
                    self.buffers[i].extend(chunk)

                self.elapsed = time.time() - self.start_time
                if self.duration > 0 and self.elapsed >= self.duration:
                    self.running = False
                    break

        finally:
            self._finalize()

    def _finalize(self):
        try:
            self.board.stop_stream()
        except Exception:
            pass

        if self.all_data:
            combined = np.concatenate(self.all_data, axis=1)
            print(f"\n[*] Total samples: {combined.shape[1]}")
            if self.output_file:
                DataFilter.write_file(combined, self.output_file, "w")
                print(f"[*] Saved to: {self.output_file}")

        try:
            self.board.release_session()
            print("[*] Session released.")
        except Exception:
            pass


# ── Real-time plot ────────────────────────────────────────────────────────────

def launch_plot(streamer, eeg_channels, sample_rate, window_size, duration):
    n_ch  = len(eeg_channels)
    t_axis = np.linspace(-WINDOW_SECONDS, 0, window_size)

    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("OpenBCI Ganglion — Live EEG", color="white", fontsize=13)

    if n_ch == 1:
        axes = [axes]

    lines = []
    for i, ax in enumerate(axes):
        ax.set_facecolor("#16213e")
        ax.set_ylabel(CH_LABELS[i], color=CH_COLORS[i], fontsize=10)
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        line, = ax.plot(t_axis, np.zeros(window_size),
                        color=CH_COLORS[i], linewidth=0.8)
        lines.append(line)

    axes[-1].set_xlabel("Time (s)", color="gray")

    # Status text in bottom-left
    status_text = fig.text(0.01, 0.01, "", color="gray", fontsize=8)

    def update(_):
        for i, line in enumerate(lines):
            y = np.array(streamer.buffers[i])
            line.set_ydata(y)
            # Auto-scale each subplot
            pad = max((y.max() - y.min()) * 0.1, 10)
            axes[i].set_ylim(y.min() - pad, y.max() + pad)

        elapsed = streamer.elapsed
        dur_str = f"{duration}s" if duration > 0 else "∞"
        status_text.set_text(f"t = {elapsed:.1f}s / {dur_str}   |   "
                             f"{sample_rate} Hz   |   {WINDOW_SECONDS}s window   |   "
                             f"BP {BANDPASS_LOW}-{BANDPASS_HIGH} Hz   notch {NOTCH_FREQ} Hz")

        if not streamer.running:
            ani.event_source.stop()
            plt.title("Stream ended", color="gray")

        return lines + [status_text]

    ani = animation.FuncAnimation(
        fig, update, interval=40, blit=False, cache_frame_data=False
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()          # blocks until window is closed
    streamer.stop()     # signal thread to stop when plot is closed


# ── Main ──────────────────────────────────────────────────────────────────────

def run(port, mac, native_ble, duration, output_file):
    BoardShim.disable_board_logger()

    board_id     = get_board_id(native_ble)
    params       = build_params(port, mac, native_ble)
    board        = BoardShim(board_id, params)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sample_rate  = BoardShim.get_sampling_rate(board_id)
    window_size  = int(sample_rate * WINDOW_SECONDS)

    print(f"[*] Board       : OpenBCI Ganglion (id={board_id})")
    print(f"[*] EEG channels: {eeg_channels}")
    print(f"[*] Sample rate : {sample_rate} Hz")
    print(f"[*] Connection  : {'Native BLE' if native_ble else 'USB dongle / ' + port}")
    print()

    try:
        print("[*] Preparing session...")
        board.prepare_session()
        print("[*] Starting stream...")
        board.start_stream(45000)
        print("[*] Plot window opened — close it to stop.\n")
    except BrainFlowError as e:
        print(f"[ERROR] {e}")
        if board.is_prepared():
            board.release_session()
        sys.exit(1)

    streamer = GanglionStreamer(board, eeg_channels, sample_rate,
                                window_size, duration, output_file)
    streamer.start()

    # Plot runs on main thread (required by matplotlib)
    launch_plot(streamer, eeg_channels, sample_rate, window_size, duration)

    streamer.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description="OpenBCI Ganglion — real-time EEG plot")
    parser.add_argument("--port",     default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--mac",      default=GANGLION_MAC)
    parser.add_argument("--native",   action="store_true", default=USE_NATIVE_BLE)
    parser.add_argument("--duration", type=int, default=STREAM_DURATION_SEC,
                        help="Stop after N seconds (0 = run until window closed)")
    parser.add_argument("--output",   default=None,
                        help="Save all data to CSV (e.g. data.csv)")
    args = parser.parse_args()

    run(args.port, args.mac, args.native, args.duration, args.output)


if __name__ == "__main__":
    main()
