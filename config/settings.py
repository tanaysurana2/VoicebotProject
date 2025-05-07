from enum import Enum

class STTProvider(str, Enum):
    FAST_RTC = "FastRTC"
    ASSEMBLY_AI = "AssemblyAI"

# Add any other configuration settings here