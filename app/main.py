from fastapi import FastAPI, UploadFile
import os
from rembg import remove
from PIL import Image
import uvicorn
import cv2
import numpy as np

app = FastAPI()
upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)
counter = 0

@app.post("/upload/")
async def upload(file: UploadFile, file2: UploadFile):
    global counter
    os.makedirs(f"{upload_dir}/clothe_set{counter}", exist_ok=True)
    file_path = f"{upload_dir}/clothe_set{counter}/file{1}.png"
    file2_path = f"{upload_dir}/clothe_set{counter}/file{2}.png"
    counter += 1
    with open(file_path, "wb") as f, open(file2_path, "wb") as f2:
        f.write(await file.read())
        f2.write(await file2.read())
    return {"upload": "ok"}

mask_dir = "masks"
os.makedirs(mask_dir, exist_ok=True)

@app.put("/make_mask/{id}")
async def remove_file(id: int):
    input_path = f"uploads/file{id}.png"
    output_path = f"masks/mask_{id}.png"

    if not os.path.exists(input_path):
        return {"error": f"File file{id} not found"}

    input_image = Image.open(input_path)
    output_image = (remove(input_image)).convert("L")
    output_image.save(output_path)

    return {"status": "ok", "mask": output_path}


@app.get("/")
def root():
    return {"message": "ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", reload = True)

    # seg = cv2.imread("masks/mask_1.png", 0)
    # # mask = np.where(seg == 5, 255, 0).astype(np.uint8)
    # # cv2.imwrite("tshirt_mask.png", mask)
    # print(np.unique(seg))
    # print(seg.shape)
