from fastapi import FastAPI

app = FastAPI(
    title="Education AI - Côte d'Ivoire",
    description="Sovereign curriculum-grounded educational AI prototype",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "project": "Education AI - Côte d'Ivoire",
        "status": "prototype",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "education-ai-api",
    }