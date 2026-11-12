import sounddevice as sd
import numpy as np

print("Devices:", sd.query_devices())
print("Default device:", sd.default.device)

print("Recording 3 seconds...")
recording = sd.rec(int(3 * 16000), samplerate=16000, channels=2, dtype='int16')
sd.wait()

print("Max amplitude Ch 1:", np.max(np.abs(recording[:, 0])))
print("Max amplitude Ch 2:", np.max(np.abs(recording[:, 1])))
