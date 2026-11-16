import sounddevice as sd
import numpy as np

def test_api(name, device_idx, channels):
    try:
        recording = sd.rec(int(1 * 16000), samplerate=16000, channels=channels, device=device_idx, dtype='int16')
        sd.wait()
        amp = np.max(np.abs(recording))
        print(f"{name} (dev {device_idx}, {channels}ch): amp={amp}")
    except Exception as e:
        print(f"{name} (dev {device_idx}, {channels}ch): FAILED - {e}")

test_api("WASAPI default", None, 2)
test_api("DirectSound dev5", 5, 2)
test_api("DirectSound dev5 4ch", 5, 4)
test_api("WASAPI dev9", 9, 2)
test_api("WASAPI dev9 1ch", 9, 1)
