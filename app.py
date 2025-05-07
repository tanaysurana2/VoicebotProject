from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes import views, interview
from utils.logging import configure_logging

configure_logging()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(views.router)
app.include_router(interview.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=7860)