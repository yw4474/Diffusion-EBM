# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.image_classifier import SimpleImageClassifier

app = FastAPI(title="CNN Classifier API")

# Instantiate a classifier (CPU is fine for now)
classifier = SimpleImageClassifier(dataset="cifar10", device="cpu")

class ClassifyRequest(BaseModel):
    image_path: str  # e.g. "sample.jpg" in your project folder

@app.get("/")
def read_root():
    return {"status": "ok", "message": "CNN Classifier API is running"}

@app.post("/classify")
def classify_image(req: ClassifyRequest):
    try:
        result = classifier.predict(req.image_path)
        return {"prediction": result}
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Image file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# app/main.py
from fastapi import FastAPI
from app.api.router import router

def create_app() -> FastAPI:
    app = FastAPI(title="GAN & CNN API")
    app.include_router(router)
    return app

app = create_app()
