# complete Working Code


from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Callable
import gradio as gr
import uuid
import numpy as np
import assemblyai as aai
from enum import Enum
import requests
from dataclasses import dataclass
from fastrtc import (
    ReplyOnPause, WebRTC, get_stt_model, get_tts_model,
    AlgoOptions, SileroVadOptions, AdditionalOutputs, KokoroTTSOptions
)
import logging
import asyncio
import time
from threading import Thread, Event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app 
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Serve static files (if you have any CSS/JS files)
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========== MODELS & CONFIGURATIONS =============
class STTProvider(str, Enum):
    FAST_RTC = "FastRTC"
    ASSEMBLY_AI = "AssemblyAI"

class InterviewPayload(BaseModel):
    resume_url: str
    job_description: str
    job_application_id: int
    job_skills: List[str]
    mandatory_skills: List[List[str]]

class InterviewRequestModel(BaseModel):
    resume_url: str
    job_description: str = ""
    job_application_id: int
    job_skills: List[str]
    mandatory_skills: List[List[str]]
    optional_skills: List[List[str]] = []

    def to_dict(self) -> dict:
        return {
            "resume_url": self.resume_url,
            "job_description": self.job_description,
            "job_application_id": self.job_application_id,
            "job_skills": self.job_skills,
            "mandatory_skills": self.mandatory_skills,
            "optional_skills": self.optional_skills
        }

@dataclass
class VoicebotConfig:
    stt_provider: STTProvider
    questions: List[str]
    tts_options: KokoroTTSOptions
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
            tts_options=KokoroTTSOptions(
                voice="af_heart",
                speed=1.0,
                lang="en-us"
            ),
            fastrtc_stt_model=get_stt_model(),
            tts_model=get_tts_model()
        )

# =========== INTERVIEW STATE MANAGEMENT =============
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

    # def get_greeting(self) -> str:
    #     next_q = self.get_next_question()
    #     if next_q is None:
    #         return "Let's begin"
    #     return "Let's begin " + next_q
    
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

        # If follow-ups are disabled, always move to next main question
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

        # Only move to next main question if followups are done
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
            # If followups are not done, don't move to next main question
            return None
    
    def store_answer(self, answer: str) -> None:
        if self.current_question:
            self.qa_pairs.append((self.current_question, answer))
            logging.info(f"Stored answer for question: {self.current_question}")
            logging.info(f"Answer: {answer}")
            self.awaiting_response = False


# =========== ASSEMBLYAI TRANSCRIBER =============
class AssemblyAITranscriber:
    def __init__(self):
        self.complete_transcript = ""
        self.partial_transcript = ""
        self.is_final = Event()
        self.transcriber = None
        self.streaming_thread = None
        self.callbacks = []
        self.final_transcripts = []
        self.has_speech = False
        self.last_activity = 0
        self.is_connected = False
        self.final_transcript_received = Event()

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

    def transcribe(self, audio: Tuple[int, np.ndarray]) -> str:
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

# =========== VOICEBOT IMPLEMENTATION =============
class InterviewVoicebot:
    def __init__(self, config: VoicebotConfig, followup_enabled: bool = True):
        self.config = config
        self.state = InterviewState(config.questions)
        self.assemblyai_transcriber = None
        if self.config.stt_provider == STTProvider.ASSEMBLY_AI and aai.settings.api_key:
            self.assemblyai_transcriber = AssemblyAITranscriber()
        self.transformers_convo: List[Dict[str, str]] = []
        self.gradio_convo: List[Tuple[str, str]] = []
        self.followup_enabled = followup_enabled
        self._transcript_buffer = {"text": "", "is_final": False}
        self._updates_queue: List[Dict] = []
        self.interview_api = interview_api
        self._current_audio_processor = None

    def startup(self):
        greeting = "Hello! Welcome to the interview. Let's begin."
        self.transformers_convo = [{"role": "assistant", "content": greeting}]
        self.gradio_convo = [(None, greeting)]
        
        yield AdditionalOutputs(self.transformers_convo, self.gradio_convo)
        
        for chunk in self.config.tts_model.stream_tts_sync(greeting, options=self.config.tts_options):
            yield chunk

        time.sleep(1)  # Pause for naturalness

        first_question = self.state.get_next_question(self.followup_enabled)
        self.transformers_convo.append({"role": "assistant", "content": first_question})
        self.gradio_convo.append((None, first_question))
        
        yield AdditionalOutputs(self.transformers_convo, self.gradio_convo)
        
        for chunk in self.config.tts_model.stream_tts_sync(first_question, options=self.config.tts_options):
            yield chunk

    def process_audio_fastrtc(self, audio: Tuple[int, np.ndarray]):
        user_response = self.config.fastrtc_stt_model.stt(audio)
        logging.info(f"Transcribed user response: {user_response}")

        if self.state.awaiting_response:
            self.state.store_answer(user_response)
            followup_generated = False

            # Only generate follow-up if enabled
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

        yield AdditionalOutputs(self.transformers_convo, self.gradio_convo)

        for chunk in self.config.tts_model.stream_tts_sync(next_question, options=self.config.tts_options):
            yield chunk

    def process_audio_assemblyai(self, audio: Tuple[int, np.ndarray]):
        # Reset state
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

        # Yield partials as they come in
        while self._updates_queue:
            partial, is_final = self._updates_queue.pop(0)
            print(f"[AssemblyAI] Yielding partial: {partial}, is_final: {is_final}")
            if partial and partial != last_partial:
                last_partial = partial
                # For partials, update the last user message
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
                yield AdditionalOutputs(self.transformers_convo, self.gradio_convo)
            if is_final:
                print(f"[AssemblyAI] Final transcript received: {partial}")
                final_transcript = partial

        # After final transcript, update the last user message (like FastRTC)
        if final_transcript and final_transcript.strip():
            print(f"[AssemblyAI] Appending final transcript to UI: {final_transcript}")
            if self.gradio_convo and self.gradio_convo[-1][0] is not None:
                self.gradio_convo[-1] = (final_transcript, None)
            else:
                self.gradio_convo.append((final_transcript, None))
            print(f"[DEBUG] Yielding to Gradio (final): gradio_convo={self.gradio_convo}")
            yield AdditionalOutputs(self.transformers_convo, self.gradio_convo)

        # Follow-up logic (same as FastRTC)
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

        yield AdditionalOutputs(self.transformers_convo, self.gradio_convo)

        for chunk in self.config.tts_model.stream_tts_sync(next_question, options=self.config.tts_options):
            yield chunk

    def process_audio(self, audio: Tuple[int, np.ndarray]):
        logging.info(f"[Voicebot] Current STT provider: {self.config.stt_provider}")
        logging.info(f"[Voicebot] AssemblyAI transcriber initialized: {self.assemblyai_transcriber is not None}")
        if self.config.stt_provider == STTProvider.FAST_RTC:
            yield from self.process_audio_fastrtc(audio)
        else:
            if not self.assemblyai_transcriber:
                if aai.settings.api_key:
                    self.assemblyai_transcriber = AssemblyAITranscriber()
                else:
                    raise ValueError("AssemblyAI API key not configured")
            yield from self.process_audio_assemblyai(audio)

    def process_audio_with_provider(self, audio, stt_provider_value):
        try:
            # Convert string value to STTProvider enum
            provider = STTProvider(stt_provider_value)
            self.config.stt_provider = provider
            
            # Initialize AssemblyAI transcriber if needed
            if provider == STTProvider.ASSEMBLY_AI and not self.assemblyai_transcriber:
                if aai.settings.api_key:
                    self.assemblyai_transcriber = AssemblyAITranscriber()
                else:
                    raise ValueError("AssemblyAI API key not configured")
            
            logging.info(f"[Voicebot] Using STT provider (from UI): {self.config.stt_provider}")
            yield from self.process_audio(audio)
        except Exception as e:
            logging.error(f"[Voicebot] Error switching STT provider: {str(e)}")
            raise

    def update_provider(self, provider: STTProvider):
        """Update the STT provider and reinitialize necessary components"""
        self.config.stt_provider = provider
        if provider == STTProvider.ASSEMBLY_AI and not self.assemblyai_transcriber:
            if aai.settings.api_key:
                self.assemblyai_transcriber = AssemblyAITranscriber()
                logging.info("[Voicebot] Initialized AssemblyAI transcriber")
            else:
                raise ValueError("AssemblyAI API key not configured")
        logging.info(f"[Voicebot] Provider updated to: {provider.value}")


# =========== GLOBAL SESSION STORAGE =============
voicebot_sessions = {}  # <-- Add this global dictionary to store InterviewVoicebot instances

# =========== API HANDLERS =============
class InterviewAPI:
    def __init__(self, config_service_root: str):
        self.config_service_root = config_service_root
        self.active_sessions: Dict[str, VoicebotConfig] = {}

    async def fetch_questions(self, request: InterviewRequestModel) -> List[str]:
        try:
            external_api_url = f"{self.config_service_root}/v2/generate_qna"
            logging.info(f"Fetching questions from: {external_api_url}")    
            logging.info(f"Request payload: {request.to_dict()}")
            response = requests.post(external_api_url, json=request.to_dict())
            response.raise_for_status()
            
            qna_data = response.json()
            print(f"qna_data: {qna_data}")
            # First handle mandatory questions
            questions = []
            for q in qna_data.get("questions", []):
                if q.get("question_text"):
                    # Log each question's details for debugging
                    logging.info(f"Processing question: {q}")
                    question_text = q["question_text"].strip()
                    is_mandatory = q.get("is_mandatory", False)
                    skill = q.get("skill", "")
                    
                    if question_text:
                        # Add mandatory questions first
                        if is_mandatory:
                            questions.insert(0, question_text)
                        else:
                            questions.append(question_text)
            
            logging.info(f"Fetched {len(questions)} questions:")
            for i, q in enumerate(questions, 1):
                logging.info(f"Question {i}: {q}")
            
            if not questions:
                logging.error("No questions fetched from API. Using default questions.")
                questions = [
                    "Can you tell me about yourself?",
                    "What are your strengths and weaknesses?",
                    "Where do you see yourself in five years?"
                ]
            
            return questions
        except Exception as e:
            logging.error(f"Error fetching questions: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch questions: {str(e)}")

    async def get_followup_question(self, question: str, answer: str) -> str:
        try:
            followup_api_url = f"{self.config_service_root}/v3/generate_followup"
            payload = {
                "question": question,
                "answer": answer
            }
            logging.info(f"Generating follow-up question for: {question}")
            logging.info(f"Previous answer: {answer}")
            logging.info(f"Follow-up API request payload: {payload}")
            
            response = requests.post(followup_api_url, json=payload)
            response.raise_for_status()
            
            followup_data = response.json()
            print(f"[DEBUG] Follow-up API raw response: {followup_data}")
            followup_question = followup_data.get("follow_up_question", "")
            
            if not followup_question:
                logging.warning("No follow-up question received from API")
                return None
                
            logging.info(f"Generated follow-up question: {followup_question}")
            return followup_question
            
        except Exception as e:
            logging.error(f"Error generating follow-up question: {str(e)}")
            return None

# Initialize API and configs storage
interview_api = InterviewAPI(config_service_root="https://ibd-dev.talent500.co")
interview_configs: Dict[str, VoicebotConfig] = {}

# =========== ROUTES =============
@app.get("/", response_class=HTMLResponse)
async def get_input_form(request: Request):
    return templates.TemplateResponse("input_form.html", {"request": request})

@app.post("/configure_interview")
async def configure_interview(request: Request):
    try:
        form_data = await request.form()
        
        # Debug logging
        logging.info("Received form data keys:")
        for key in form_data.keys():
            logging.info(f"Key: {key}, Value: {form_data[key]}")
        
        # Extract and process job skills using getlist()
        job_skills = []
        if 'jobSkills[]' in form_data:
            job_skills = [skill.strip() for skill in form_data.getlist('jobSkills[]') if skill.strip()]
            for skill in job_skills:
                logging.info(f"Added job skill: {skill}")

        # Extract and process mandatory skills
        mandatory_skills = []
        mandatory_group = []
        for key in form_data.keys():
            if key.startswith('mandatorySkills[') and key.endswith('][]'):
                if form_data[key].strip():
                    mandatory_group.append(form_data[key].strip())
            elif key.startswith('mandatorySkills[') and '][]' in key:
                if mandatory_group:
                    mandatory_skills.append(mandatory_group)
                    mandatory_group = []

        if mandatory_group:
            mandatory_skills.append(mandatory_group)

        # Create the payload
        payload = InterviewPayload(
            resume_url=form_data.get('resume_url'),
            job_description=form_data.get('job_description', ''),
            job_application_id=int(form_data.get('job_application_id')),
            job_skills=job_skills,
            mandatory_skills=mandatory_skills
        )

        # Get the followup_enabled flag (checkbox returns 'on' if checked)
        followup_enabled = form_data.get('followup_enabled') == 'on'

        # Get the interrupt_enabled flag (checkbox returns 'on' if checked)
        interrupt_enabled = form_data.get('interruptToggle') == 'on'

        # Get the STT provider
        stt_provider = form_data.get('stt_provider', 'FastRTC')
        if stt_provider == 'AssemblyAI' and not aai.settings.api_key:
            raise HTTPException(
                status_code=400,
                detail="AssemblyAI API key not configured. Please use FastRTC instead."
            )

        # Log the values for debugging
        logging.info(f"######################### Follow-up enabled: {followup_enabled}")
        logging.info(f"######################### Interrupt enabled: {interrupt_enabled}")
        logging.info(f"######################### STT Provider: {stt_provider}")

        # Generate a session ID
        session_id = str(uuid.uuid4())

        # Store the payload and configuration
        interview_configs[session_id] = {
            "payload": payload.model_dump(),
            "stt_provider": stt_provider,
            "tts_voice": "af_heart",
            "tts_speed": 1.0,
            "tts_lang": "en-us",
            "followup_enabled": followup_enabled,
            "interrupt_enabled": interrupt_enabled
        }

        # Log before fetching questions
        logging.info(f"Fetching questions for session {session_id}")
        questions = await interview_api.fetch_questions(InterviewRequestModel(**payload.dict()))
        logging.info(f"Storing {len(questions)} questions for session {session_id}")
        interview_configs[session_id]["questions"] = questions

        # Redirect to the interview UI
        return RedirectResponse(
            url=f"/interview_ui?session_id={session_id}",
            status_code=303
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_interview")
async def start_interview(session_id: str = Form(...)):
    config_dict = interview_configs.get(session_id)
    if not config_dict:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    # Log the questions being used
    questions = config_dict["questions"]
    logging.info(f"Starting interview for session {session_id}")
    logging.info(f"Using {len(questions)} questions:")
    for i, q in enumerate(questions, 1):
        logging.info(f"Question {i}: {q}")
    
    # Create a proper VoicebotConfig object
    config = VoicebotConfig.create_default(questions=questions)
    config.stt_provider = STTProvider(config_dict["stt_provider"])
    config.tts_options = KokoroTTSOptions(
        voice=config_dict["tts_voice"],
        speed=config_dict["tts_speed"],
        lang=config_dict["tts_lang"]
    )
    # Pass followup_enabled to InterviewVoicebot
    followup_enabled = config_dict.get("followup_enabled", True)
    interrupt_enabled = config_dict.get("interrupt_enabled", False)

    # Log the values for debugging
    logging.info(f"######################### Starting interview with followup_enabled={followup_enabled}, interrupt_enabled={interrupt_enabled}")

    voicebot = InterviewVoicebot(config, followup_enabled=followup_enabled)
    voicebot_sessions[session_id] = voicebot  # <-- Store the InterviewVoicebot instance for this session

    demo = create_ui(session_id, interrupt_enabled=interrupt_enabled)
    
    # Mount the Gradio app at a unique route
    gradio_route = f"/interview/{session_id}"
    app.mount(gradio_route, gr.mount_gradio_app(app=app, blocks=demo, path=gradio_route))
    
    # Redirect to the mounted Gradio app
    return RedirectResponse(url=gradio_route, status_code=303)

@app.get("/interview_ui")
async def interview_ui(request: Request, session_id: str):
    if session_id not in interview_configs:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    return templates.TemplateResponse(
        "interview_ui.html",
        {
            "request": request,
            "session_id": session_id
        }
    )

# =========== GRADIO UI =============
def get_audio_processor(session_id):
    def process_audio(audio):
        if session_id in voicebot_sessions:
            voicebot = voicebot_sessions[session_id]
            logging.info(f"[AudioProcessor] Current STT provider: {voicebot.config.stt_provider}")
            return voicebot.process_audio(audio)
        else:
            raise ValueError(f"No voicebot found for session {session_id}")
    return process_audio

def create_ui(session_id: str, interrupt_enabled: bool = False):
    with gr.Blocks() as demo:
        gr.HTML(
            f"""
            <h1 style='text-align: center'>
            Job Interview Simulation
            </h1>
            <p style='text-align: center'>
            {len(interview_configs[session_id]['questions'])} questions prepared for this interview
            </p>
            """
        )

        status = gr.Textbox(label="Status", visible=False)
        transformers_convo = gr.State([])
        shared_session_id = gr.State(session_id)

        logging.info(f"[GradioUI] Initial STT provider: {interview_configs[session_id]['stt_provider']}")

        with gr.Row():
            with gr.Column():
                audio = WebRTC(
                    label="Interview Stream",
                    mode="send-receive",
                    modality="audio"
                )
            with gr.Column():
                transcript = gr.Chatbot(label="Interview Transcript", height=500)

        def update_stt_provider(provider_value: str, shared_session_id):
            try:
                provider = STTProvider(provider_value)
                if shared_session_id in voicebot_sessions:
                    voicebot = voicebot_sessions[shared_session_id]
                    voicebot.update_provider(provider)
                    return f"STT provider set to {provider.value}", gr.update(visible=False), shared_session_id
                else:
                    logging.warning(f"[GradioUI] No running voicebot for session {shared_session_id}")
                    return f"STT provider set to {provider.value}", gr.update(visible=False), shared_session_id
            except Exception as e:
                logging.error(f"[GradioUI] Error setting STT provider: {str(e)}")
                return f"Error: {str(e)}", gr.update(visible=True), shared_session_id

        stt_provider = gr.Radio(
            choices=[p.value for p in STTProvider],
            value=interview_configs[session_id]['stt_provider'],
            label="Speech-to-Text Provider"
        )

        stt_provider.change(
            update_stt_provider,
            inputs=[stt_provider, shared_session_id],
            outputs=[status, status, shared_session_id]
        )

        # --- FIX: Use ReplyOnPause for audio.stream ---
        audio.stream(
            ReplyOnPause(
                get_audio_processor(session_id),
                can_interrupt=interrupt_enabled,
                startup_fn=voicebot_sessions[session_id].startup,
                algo_options=AlgoOptions(
                    audio_chunk_duration=0.6,
                    started_talking_threshold=0.2,
                    speech_threshold=0.1
                ),
                model_options=SileroVadOptions(
                    threshold=0.5,
                    min_speech_duration_ms=100,
                    min_silence_duration_ms=1000
                )
            ),
            inputs=[audio],
            outputs=[audio],
            time_limit=180
        )

        audio.on_additional_outputs(
            lambda s, a: (s, a),
            outputs=[transformers_convo, transcript],
            queue=False,
            show_progress="hidden"
        )

    return demo


# =========== MAIN =============
if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="localhost", port=7860)