from pathlib import Path
import soundfile as sf

def wav_metadata(path: Path) -> dict:
    info = sf.info(str(path))
    duration = info.frames / float(info.samplerate)
    return {
        "sample_rate": float(info.samplerate),
        "channels": int(info.channels),
        "duration_sec": float(duration),
    }
