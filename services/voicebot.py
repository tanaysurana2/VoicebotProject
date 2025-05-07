import logging
import numpy as np
import asyncio
import time
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from models.base_models import InterviewRequestModel
from config.settings import STTProvider
from services.stt_service import AssemblyAITranscriber

@dataclass
class VoicebotConfig:
    stt_provider: STTProvider
    questions: List[str]
    tts_options: dict
    fastrtc_stt_model: any
    tts_model: any

    @classmethod
    def create_default(cls, questions: Optional[List[str]] = None) -> 'VoicebotConfig':
        default_questions = questions or [
            "Can you tell me about yourself?",
            "What are your strengths and weaknesses?",
            "Where do you see yourself in five years?"
        ]
        
        return cls(
            stt_provider=STTProvider.FAST_RTC,
            questions=default_questions,
            tts_options={
                "voice": "af_heart",
                "speed": 1.0,
                "lang": "en-us"
            },
            fastrtc_stt_model=None,  # Will be set later
            tts_model=None  # Will be set later
        )

class InterviewState:
    def __init__(self, questions: List[str]):
        self.questions = questions
        self.current_index = 0
        self.qa_pairs: List[Tuple[str, str]] = []
        self.started = False
        self.awaiting_response = False
        self.current_question = ""
        self.pending_followup = None
        self.followup_count = 0
        logging.info(f"Interview state initialized with {len(questions)} questions")
    
    def can_ask_followup(self) -> bool:
        return self.followup_count < 2

    def increment_followup(self):
        self.followup_count += 1

    def reset_followup(self):
        self.followup_count = 0

    def get_next_question(self, followup_enabled=True) -> str:
        if self.pending_followup:
            self.current_question = self.pending_followup
            self.pending_followup = None
            self.awaiting_response = True
            self.increment_followup()
            return self.current_question

        if not self.started:
            self.started = True
            self.current_question = self.questions[0]
            self.awaiting_response = True
            self.reset_followup()
            return self.current_question

        if not followup_enabled:
            self.current_index += 1
            if self.current_index < len(self.questions):
                self.current_question = self.questions[self.current_index]
                self.awaiting_response = True
                self.reset_followup()
                return self.current_question
            else:
                self.started = False
                return "Thank you for your time. This concludes the interview."

        if self.followup_count >= 2:
            self.current_index += 1
            if self.current_index < len(self.questions):
                self.current_question = self.questions[self.current_index]
                self.awaiting_response = True
                self.reset_followup()
                return self.current_question
            else:
                self.started = False
                return "Thank you for your time. This concludes the interview."
        else:
            return None
    
    def store_answer(self, answer: str) -> None:
        if self.current_question:
            self.qa_pairs.append((self.current_question, answer))
            logging.info(f"Stored answer for question: {self.current_question}")
            logging.info(f"Answer: {answer}")
            self.awaiting_response = False

class InterviewVoicebot:
    def __init__(self, config: VoicebotConfig, followup_enabled: bool = True):
        self.config = config
        self.state = InterviewState(config.questions)
        self.assemblyai_transcriber = None
        if self.config.stt_provider == STTProvider.ASSEMBLY_AI:
            self.assemblyai_transcriber = AssemblyAITranscriber()
        self.transformers_convo: List[Dict[str, str]] = []
        self.gradio_convo: List[Tuple[str, str]] = []
        self.followup_enabled = followup_enabled
        self._transcript_buffer = {"text": "", "is_final": False}
        self._updates_queue: List[Dict] = []
        self._current_audio_processor = None

    def startup(self):
        greeting = "Hello! Welcome to the interview. Let's begin."
        self.transformers_convo = [{"role": "assistant", "content": greeting}]
        self.gradio_convo = [(None, greeting)]
        
        yield {"transformers_convo": self.transformers_convo, "gradio_convo": self.gradio_convo}
        
        for chunk in self.config.tts_model.stream_tts_sync(greeting, options=self.config.tts_options):
            yield chunk

        time.sleep(1)

        first_question = self.state.get_next_question(self.followup_enabled)
        self.transformers_convo.append({"role": "assistant", "content": first_question})
        self.gradio_convo.append((None, first_question))
        
        yield {"transformers_convo": self.transformers_convo, "gradio_convo": self.gradio_convo}
        
        for chunk in self.config.tts_model.stream_tts_sync(first_question, options=self.config.tts_options):
            yield chunk

    def process_audio_fastrtc(self, audio: Tuple[int, np.ndarray]):
        user_response = self.config.fastrtc_stt_model.stt(audio)
        logging.info(f"Transcribed user response: {user_response}")

        if self.state.awaiting_response:
            self.state.store_answer(user_response)
            followup_generated = False

            if self.followup_enabled and self.state.can_ask_followup():
                try:
                    followup = asyncio.run(self.interview_api.get_followup_question(
                        self.state.current_question,
                        user_response
                    ))
                    if followup and followup.strip() and followup != self.state.current_question:
                        self.state.pending_followup = followup
                        followup_generated = True
                except Exception as e:
                    logging.error(f"Error generating follow-up: {str(e)}")
                    pass

            if not followup_generated:
                self.state.awaiting_response = False

        next_question = self.state.get_next_question(self.followup_enabled)
        if next_question is None:
            return

        self.transformers_convo.append({"role": "user", "content": user_response})
        self.transformers_convo.append({"role": "assistant", "content": next_question})

        self.gradio_convo.append((user_response, None))
        self.gradio_convo.append((None, next_question))

        yield {"transformers_convo": self.transformers_convo, "gradio_convo": self.gradio_convo}

        for chunk in self.config.tts_model.stream_tts_sync(next_question, options=self.config.tts_options):
            yield chunk

    def process_audio_assemblyai(self, audio: Tuple[int, np.ndarray]):
        self._transcript_buffer = {"text": "", "is_final": False}
        self._updates_queue = []

        def debug_callback(transcript, is_final):
            print(f"[DEBUG] Callback received: transcript='{transcript}', is_final={is_final}")
            self._updates_queue.append((transcript, is_final))

        self.assemblyai_transcriber.clear_callbacks()
        self.assemblyai_transcriber.add_callback(debug_callback)
        print("[AssemblyAI] Starting transcription")
        transcript = self.assemblyai_transcriber.transcribe(audio)
        print(f"[AssemblyAI] Transcription complete: {transcript}")

        last_partial = ""
        final_transcript = None

        while self._updates_queue:
            partial, is_final = self._updates_queue.pop(0)
            print(f"[AssemblyAI] Yielding partial: {partial}, is_final: {is_final}")
            if partial and partial != last_partial:
                last_partial = partial
                print(f"[AssemblyAI] Updating UI with partial: {partial}")
                if self.gradio_convo and self.gradio_convo[-1][0] is not None:
                    self.gradio_convo[-1] = (partial, None)
                else:
                    self.gradio_convo.append((partial, None))
                if self.transformers_convo and self.transformers_convo[-1]["role"] == "user":
                    self.transformers_convo[-1]["content"] = partial
                else:
                    self.transformers_convo.append({"role": "user", "content": partial})
                print(f"[DEBUG] Yielding to Gradio (partial): gradio_convo={self.gradio_convo}")
                yield {"transformers_convo": self.transformers_convo, "gradio_convo": self.gradio_convo}
            if is_final:
                print(f"[AssemblyAI] Final transcript received: {partial}")
                final_transcript = partial

        if final_transcript and final_transcript.strip():
            print(f"[AssemblyAI] Appending final transcript to UI: {final_transcript}")
            if self.gradio_convo and self.gradio_convo[-1][0] is not None:
                self.gradio_convo[-1] = (final_transcript, None)
            else:
                self.gradio_convo.append((final_transcript, None))
            print(f"[DEBUG] Yielding to Gradio (final): gradio_convo={self.gradio_convo}")
            yield {"transformers_convo": self.transformers_convo, "gradio_convo": self.gradio_convo}

        if self.state.awaiting_response:
            print(f"[AssemblyAI] Storing answer: {final_transcript or transcript}")
            self.state.store_answer(final_transcript or transcript)
            followup_generated = False

            if self.followup_enabled and self.state.can_ask_followup():
                try:
                    print(f"[AssemblyAI] Requesting follow-up for Q: {self.state.current_question}, A: {final_transcript or transcript}")
                    followup = asyncio.run(self.interview_api.get_followup_question(
                        self.state.current_question,
                        final_transcript or transcript
                    ))
                    print(f"[AssemblyAI] Follow-up received: {followup}")
                    if followup and followup.strip() and followup != self.state.current_question:
                        self.state.pending_followup = followup
                        followup_generated = True
                except Exception as e:
                    print(f"[AssemblyAI] Error generating follow-up: {str(e)}")
                    pass

            if not followup_generated:
                self.state.awaiting_response = False

        next_question = self.state.get_next_question(self.followup_enabled)
        print(f"[AssemblyAI] Next question: {next_question}")
        self.transformers_convo.append({"role": "assistant", "content": next_question})
        self.gradio_convo.append((None, next_question))

        yield {"transformers_convo": self.transformers_convo, "gradio_convo": self.gradio_convo}

        for chunk in self.config.tts_model.stream_tts_sync(next_question, options=self.config.tts_options):
            yield chunk

    def process_audio(self, audio: Tuple[int, np.ndarray]):
        logging.info(f"[Voicebot] Current STT provider: {self.config.stt_provider}")
        logging.info(f"[Voicebot] AssemblyAI transcriber initialized: {self.assemblyai_transcriber is not None}")
        if self.config.stt_provider == STTProvider.FAST_RTC:
            yield from self.process_audio_fastrtc(audio)
        else:
            if not self.assemblyai_transcriber:
                raise ValueError("AssemblyAI API key not configured")
            yield from self.process_audio_assemblyai(audio)

    def process_audio_with_provider(self, audio, stt_provider_value):
        try:
            provider = STTProvider(stt_provider_value)
            self.config.stt_provider = provider
            
            if provider == STTProvider.ASSEMBLY_AI and not self.assemblyai_transcriber:
                raise ValueError("AssemblyAI API key not configured")
            
            logging.info(f"[Voicebot] Using STT provider (from UI): {self.config.stt_provider}")
            yield from self.process_audio(audio)
        except Exception as e:
            logging.error(f"[Voicebot] Error switching STT provider: {str(e)}")
            raise

    def update_provider(self, provider: STTProvider):
        self.config.stt_provider = provider
        if provider == STTProvider.ASSEMBLY_AI and not self.assemblyai_transcriber:
            raise ValueError("AssemblyAI API key not configured")
        logging.info(f"[Voicebot] Provider updated to: {provider.value}")