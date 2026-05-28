from fastapi import FastAPI, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image

from database import init_db, get_connection
from pose import estimate_shoulder_width
from mesh import generate_3d_model
from clothing_segment import segment_clothing

import os
import cv2
import numpy as np
import uvicorn
# from ai.triposr_engine import TripoEngine

app = FastAPI()

# tripo_engine = TripoEngine()

app.mount("/models", StaticFiles(directory="models"), name="models")

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# INIT
# =========================

init_db()

upload_dir = "uploads"
mask_dir = "masks"
model_dir = "models"

os.makedirs(upload_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

counter = 1

# =========================
# UPLOAD
# =========================

@app.post("/upload/")
async def upload(file: UploadFile, file2: UploadFile):

    global counter

    folder_path = f"{upload_dir}/clothe_set{counter}"
    mask_folder = f"{mask_dir}/clothe_set{counter}"

    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    # images
    front_image_path = f"{folder_path}/front.png"
    side_image_path = f"{folder_path}/side.png"

    with open(front_image_path, "wb") as f:
        f.write(await file.read())

    with open(side_image_path, "wb") as f:
        f.write(await file2.read())

    # masks
    front_mask_path = f"{mask_folder}/front_mask.png"
    side_mask_path = f"{mask_folder}/side_mask.png"

    front_image = Image.open(front_image_path)
    side_image = Image.open(side_image_path)


    # database
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO clothing_sets (folder_path)
        VALUES (?)
        """,
        (folder_path,)
    )

    clothing_set_id = cur.lastrowid

    # front image
    cur.execute(
        """
        INSERT INTO images (
            clothing_set_id,
            image_path,
            image_type
        )
        VALUES (?, ?, ?)
        """,
        (
            clothing_set_id,
            front_image_path,
            "front"
        )
    )

    # side image
    cur.execute(
        """
        INSERT INTO images (
            clothing_set_id,
            image_path,
            image_type
        )
        VALUES (?, ?, ?)
        """,
        (
            clothing_set_id,
            side_image_path,
            "side"
        )
    )

    # front mask
    cur.execute(
        """
        INSERT INTO masks (
            clothing_set_id,
            mask_path,
            mask_type
        )
        VALUES (?, ?, ?)
        """,
        (
            clothing_set_id,
            front_mask_path,
            "front"
        )
    )

    # side mask
    cur.execute(
        """
        INSERT INTO masks (
            clothing_set_id,
            mask_path,
            mask_type
        )
        VALUES (?, ?, ?)
        """,
        (
            clothing_set_id,
            side_mask_path,
            "side"
        )
    )

    conn.commit()
    conn.close()

    counter += 1

    return {
        "status": "ok",
        "id": clothing_set_id,
        "front_image": front_image_path,
        "side_image": side_image_path,
        "front_mask": front_mask_path,
        "side_mask": side_mask_path
    }

# =========================
# SELECT CLOTHING
# =========================

@app.post("/select_clothing/{id}")
async def select_clothing(
    id: int,
    x: int = Query(...),
    y: int = Query(...),
    type: str = Query(...)
):

    front_path = f"{upload_dir}/clothe_set{id}/front.png"
    side_path = f"{upload_dir}/clothe_set{id}/side.png"

    save_dir = f"{mask_dir}/clothe_set{id}"
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # FRONT
    # =========================
    if type == "front":
        ok = segment_clothing(
            front_path,
            x,
            y,
            f"{save_dir}/front_selected.png"
        )
    else:
        ok = segment_clothing(
            side_path,
            x,
            y,
            f"{save_dir}/side_selected.png"
        )

    if not ok:
        return {"error": "Clothing not detected"}

    return {
        "status": "ok",
        "type": type,
        "selected_mask": f"{save_dir}/{type}_selected.png"
    }
# =========================
# MEASURE
# =========================

@app.post("/measure/{id}")
async def measure(id: int):

    image_path = f"{upload_dir}/clothe_set{id}/front.png"

    if not os.path.exists(image_path):
        return {"error": "Image not found"}

    shoulder_px = estimate_shoulder_width(image_path)

    if shoulder_px is None:
        return {"error": "Pose not detected"}

    # примерный масштаб
    scale = 45 / shoulder_px

    mask_path = f"{mask_dir}/clothe_set{id}/front_selected.png"

    if not os.path.exists(mask_path):
        return {"error": "Selected clothing not found"}

    mask = cv2.imread(mask_path, 0)

    if mask is None:
        return {"error": "Cannot read selected mask"}

    coords = cv2.findNonZero(mask)

    if coords is None:
        return {"error": "No clothing detected"}

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"error": "No contours found"}

    largest = max(contours, key=cv2.contourArea)

    # реальные размеры формы
    area = cv2.contourArea(largest)
    x, y, w, h = cv2.approxPolyDP(largest, 0.01 * perimeter, True)

    # более “умная” оценка размеров
    perimeter = cv2.arcLength(largest, True)

    clothing_width = round(w * scale, 2)
    clothing_height = round(h * scale, 2)

    shoulder_cm = round(shoulder_px * scale, 2)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO measurements (
            clothing_set_id,
            shoulder_width,
            clothing_width,
            clothing_height
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            id,
            shoulder_cm,
            clothing_width,
            clothing_height
        )
    )

    conn.commit()
    conn.close()

    return {
        "shoulder_width_cm": shoulder_cm,
        "clothing_width_cm": clothing_width,
        "clothing_height_cm": clothing_height
    }

# =========================
# GENERATE 3D MODEL
# =========================

@app.post("/generate_3d/{id}")
async def generate_3d(id: int):

    front_mask = f"{mask_dir}/clothe_set{id}/front_selected.png"
    side_mask = f"{mask_dir}/clothe_set{id}/side_selected.png"

    if not os.path.exists(front_mask):
        return {"error": "front mask missing"}

    if not os.path.exists(side_mask):
        return {"error": "side mask missing"}

    output_path = f"{model_dir}/clothe_set{id}.obj"

    result = generate_3d_model(
        front_mask,
        side_mask,
        output_path
    )

    if result is None:
        return {"error": "generation failed"}

    return {
        "status": "ok",
        "model_url": f"/models/clothe_set{id}.obj"
    }

# =========================
# GET SELECTED MASK
# =========================

@app.get("/get_selected/{id}")
def get_selected(
    id: int,
    type: str
):

    path = f"{mask_dir}/clothe_set{id}/{type}_selected.png"

    if not os.path.exists(path):
        return {"error": "not found"}

    return FileResponse(path)

# =========================
# GET MODEL
# =========================

# @app.get("/get_model/{id}")
# def get_model(id: int):

#     path = f"models/clothe_set{id}.obj"

#     if not os.path.exists(path):
#         return {"error": "model not found"}

#     return FileResponse(
#         path,
#         media_type="model/obj"
#     )
# =========================
# ROOT
# =========================

@app.get("/")
def root():
    return {"message": "ok"}

# =========================
# RUN
# =========================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
