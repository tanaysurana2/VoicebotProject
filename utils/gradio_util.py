import gradio as gr
from fastapi import FastAPI

def mount_gradio_interface(app: FastAPI, session_id: str, gradio_app: gr.Blocks):
    gradio_route = f"/interview/{session_id}"
    app.mount(gradio_route, gr.mount_gradio_app(app=app, blocks=gradio_app, path=gradio_route))
    return gradio_route