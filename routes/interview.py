from fastapi import APIRouter, FastAPI, Request, Form, HTTPException , Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import uuid
import logging
from models.base_models import InterviewPayload, InterviewRequestModel
from services.interview_api import InterviewAPI
from services.voicebot import InterviewVoicebot, VoicebotConfig
import gradio as gr
from config.settings import STTProvider 
from utils.gradio_util import mount_gradio_interface
from fastrtc import (
    ReplyOnPause, WebRTC, get_stt_model, get_tts_model,
    AlgoOptions, SileroVadOptions, AdditionalOutputs, KokoroTTSOptions
)


router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Initialize API
interview_api = InterviewAPI(config_service_root="https://ibd-dev.talent500.co")

# Global session storage
voicebot_sessions = {}
interview_configs = {}

def get_app(request: Request) -> FastAPI:
    return request.app

@router.post("/configure_interview")
async def configure_interview(request: Request):
    try:
        form_data = await request.form()
        
        logging.info("Received form data keys:")
        for key in form_data.keys():
            logging.info(f"Key: {key}, Value: {form_data[key]}")
        
        job_skills = []
        if 'jobSkills[]' in form_data:
            job_skills = [skill.strip() for skill in form_data.getlist('jobSkills[]') if skill.strip()]
            for skill in job_skills:
                logging.info(f"Added job skill: {skill}")

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

        payload = InterviewPayload(
            resume_url=form_data.get('resume_url'),
            job_description=form_data.get('job_description', ''),
            job_application_id=int(form_data.get('job_application_id')),
            job_skills=job_skills,
            mandatory_skills=mandatory_skills
        )

        followup_enabled = form_data.get('followup_enabled') == 'on'
        interrupt_enabled = form_data.get('interruptToggle') == 'on'
        stt_provider = form_data.get('stt_provider', 'FastRTC')

        logging.info(f"######################### Follow-up enabled: {followup_enabled}")
        logging.info(f"######################### Interrupt enabled: {interrupt_enabled}")
        logging.info(f"######################### STT Provider: {stt_provider}")

        session_id = str(uuid.uuid4())

        interview_configs[session_id] = {
            "payload": payload.model_dump(),
            "stt_provider": stt_provider,
            "tts_voice": "af_heart",
            "tts_speed": 1.0,
            "tts_lang": "en-us",
            "followup_enabled": followup_enabled,
            "interrupt_enabled": interrupt_enabled
        }

        logging.info(f"Fetching questions for session {session_id}")
        questions = await interview_api.fetch_questions(InterviewRequestModel(**payload.dict()))
        logging.info(f"Storing {len(questions)} questions for session {session_id}")
        interview_configs[session_id]["questions"] = questions

        return RedirectResponse(
            url=f"/interview_ui?session_id={session_id}",
            status_code=303
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start_interview")
async def start_interview(
    request: Request,
    session_id: str = Form(...),
    app: FastAPI = Depends(get_app)
):
    config_dict = interview_configs.get(session_id)
    if not config_dict:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    questions = config_dict["questions"]
    logging.info(f"Starting interview for session {session_id}")
    logging.info(f"Using {len(questions)} questions:")
    for i, q in enumerate(questions, 1):
        logging.info(f"Question {i}: {q}")
    
    config = VoicebotConfig.create_default(questions=questions)
    config.stt_provider = STTProvider(config_dict["stt_provider"])
    config.tts_options = KokoroTTSOptions(
        voice=config_dict["tts_voice"],
        speed=config_dict["tts_speed"],
        lang=config_dict["tts_lang"]
    )
    
    followup_enabled = config_dict.get("followup_enabled", True)
    interrupt_enabled = config_dict.get("interrupt_enabled", False)

    logging.info(f"######################### Starting interview with followup_enabled={followup_enabled}, interrupt_enabled={interrupt_enabled}")

    voicebot = InterviewVoicebot(config, followup_enabled=followup_enabled)
    voicebot_sessions[session_id] = voicebot

    demo = create_ui(session_id, interrupt_enabled=interrupt_enabled)
    gradio_route = mount_gradio_interface(app, session_id, demo)
    
    return RedirectResponse(url=gradio_route, status_code=303)

@router.get("/interview_ui")
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

def mount_gradio_interface(app: FastAPI, session_id: str, demo: gr.Blocks) -> str:
    gradio_route = f"/interview/{session_id}"
    app.mount(gradio_route, gr.mount_gradio_app(app=app, blocks=demo, path=gradio_route))
    return gradio_route