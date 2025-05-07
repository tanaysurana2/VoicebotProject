import assemblyai as aai
import time
from threading import Thread, Event
from typing import Callable, Optional
from dataclasses import dataclass

@dataclass
class AssemblyAITranscriber:
    complete_transcript: str = ""
    partial_transcript: str = ""
    is_final: Event = Event()
    transcriber: Optional[any] = None
    streaming_thread: Optional[Thread] = None
    callbacks: list = None
    final_transcripts: list = None
    has_speech: bool = False
    last_activity: float = 0
    is_connected: bool = False
    final_transcript_received: Event = Event()

    def __post_init__(self):
        self.callbacks = []
        self.final_transcripts = []

    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)

    def clear_callbacks(self):
        self.callbacks = []

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.final_transcripts.append(transcript.text)
            self.has_speech = True
            self.last_activity = time.time()
            self.complete_transcript = " ".join(self.final_transcripts)
            
            for callback in self.callbacks:
                callback(self.complete_transcript, False)

        elif isinstance(transcript, aai.RealtimePartialTranscript):
            self.partial_transcript = transcript.text
            self.has_speech = True
            self.last_activity = time.time()
            
            current_transcript = self.complete_transcript + " " + self.partial_transcript if self.complete_transcript else self.partial_transcript
            current_transcript = current_transcript.strip()
            
            for callback in self.callbacks:
                callback(current_transcript, False)

    def check_silence(self):
        silence_threshold = 3.0

        while self.transcriber and self.is_connected and not self.is_final.is_set():
            current_time = time.time()
            if self.has_speech and (current_time - self.last_activity) > silence_threshold:
                print("Detected end of speech due to silence")
                self.is_final.set()
                for callback in self.callbacks:
                    callback(self.complete_transcript, True)
                self.final_transcript_received.set()
                break
            time.sleep(0.2)

    def on_error(self, error: aai.RealtimeError):
        print(f"Error: {error}")
        self.is_connected = False
        self.is_final.set()
        self.final_transcript_received.set()
        if self.transcriber:
            try:
                self.transcriber.close()
            except Exception as e:
                print(f"Error closing transcriber: {e}")

    def on_close(self):
        print("Session closed")
        self.is_connected = False
        self.is_final.set()
        for callback in self.callbacks:
            callback(self.complete_transcript, True)
        self.final_transcript_received.set()

    def stream_audio(self, audio_bytes: bytes, sample_rate: int):
        try:
            self.transcriber = aai.RealtimeTranscriber(
                sample_rate=sample_rate,
                on_data=self.on_data,
                on_error=self.on_error,
                on_close=self.on_close
            )

            self.transcriber.connect()
            self.is_connected = True

            silence_thread = Thread(target=self.check_silence)
            silence_thread.daemon = True
            silence_thread.start()

            chunk_size = 2000
            position = 0
            while position < len(audio_bytes) and not self.is_final.is_set() and self.is_connected:
                chunk = audio_bytes[position:position + chunk_size]
                self.transcriber.stream(chunk)
                position += chunk_size
                time.sleep(0.010)

            if position >= len(audio_bytes) and self.is_connected:
                self.is_final.wait(timeout=3.0)

        except Exception as e:
            print(f"Error streaming audio: {e}")
            self.is_connected = False
            self.is_final.set()
            self.final_transcript_received.set()
        finally:
            self.is_connected = False
            if self.transcriber:
                try:
                    self.transcriber.close()
                except Exception as e:
                    print(f"Error in final closing: {e}")

    def transcribe(self, audio: tuple) -> str:
        self.complete_transcript = ""
        self.partial_transcript = ""
        self.final_transcripts = []
        self.has_speech = False
        self.last_activity = time.time()
        self.is_final.clear()
        self.final_transcript_received.clear()
        self.is_connected = False

        sample_rate, audio_data = audio
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)

        audio_bytes = audio_data.tobytes()

        self.streaming_thread = Thread(target=self.stream_audio, args=(audio_bytes, sample_rate))
        self.streaming_thread.daemon = True
        self.streaming_thread.start()

        print("Waiting for final transcript...")
        self.final_transcript_received.wait(timeout=10.0)
        print(f"Final transcript received or timeout. Final: {self.complete_transcript}")

        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=1.0)

        clean_transcript = " ".join(self.complete_transcript.split())
        return clean_transcript if clean_transcript else self.partial_transcript