from enum import Enum

class AlgoOptions(str, Enum):
    AUDIO_CHUNK_DURATION = "audio_chunk_duration"
    STARTED_TALKING_THRESHOLD = "started_talking_threshold"
    SPEECH_THRESHOLD = "speech_threshold"