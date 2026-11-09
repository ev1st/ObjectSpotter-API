import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8m.pt")

def run_detection(image: Image.Image, conf: float):
    result = model.predict(source=image, conf=conf, verbose=False)[0]
    return result


@app.get("/")
async def read_root():
    return FileResponse("index.html")


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.30):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Нужен файл-изображение")

    contents = await file.read()
    try:
        photo = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Не удалось прочитать изображение")

    result = run_detection(image=photo, conf=conf)

    detections = []
    for box in result.boxes:
        detections.append({
            "label": result.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "box": [round(v, 1) for v in box.xyxy[0].tolist()],
        })
    return {"count": len(detections), "detections": detections}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)