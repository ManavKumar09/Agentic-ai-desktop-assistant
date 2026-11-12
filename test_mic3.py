import sounddevice as sd
import numpy as np

def test_api(name, device_idx, channels):
    try:
        recording = sd.rec(int(1 * 48000), samplerate=48000, channels=channels, device=device_idx, dtype='int16')
        sd.wait()
        amp = np.max(np.abs(recording))
        print(f"{name} (dev {device_idx}, {channels}ch, 48000Hz): amp={amp}")
    except Exception as e:
        print(f"{name} (dev {device_idx}, {channels}ch): FAILED - {e}")

test_api("WASAPI default", None, 2)
test_api("DirectSound dev5", 5, 2)
test_api("WASAPI dev9", 9, 2)
test_api("MME dev1 4ch", 1, 4)
