from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.model_service import train_model, classify_image, list_models

app = FastAPI(
    title="API de Clasificación de Imágenes",
    description="Entrena y clasifica imágenes con Regresión Logística sobre histogramas RGB.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/models")
def get_models():
    return {"models": list_models()}


@app.post("/train")
async def train(
    file: UploadFile = File(...),
    classifier_name: str = Form(...),
):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .zip")
    try:
        return JSONResponse(content=train_model(await file.read(), classifier_name))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    classifier_name: str = Form(...),
):
    try:
        return JSONResponse(content=classify_image(await file.read(), classifier_name))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
