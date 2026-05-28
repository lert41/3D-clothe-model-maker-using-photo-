from ultralytics import SAM
import cv2
import numpy as np

model = SAM("sam_b.pt")  # или sam_l.pt (лучше качество)

def segment_clothing(image_path, click_x, click_y, save_path):

    image = cv2.imread(image_path)
    if image is None:
        return False

    h, w, _ = image.shape

    # SAM ожидает точки в формате [[x, y]]
    results = model(image, points=[[click_x, click_y]])

    if not results or results[0].masks is None:
        return False

    masks = results[0].masks.data.cpu().numpy()

    best_mask = None
    best_area = 0

    for mask in masks:
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w, h))

        area = cv2.countNonZero(mask)

        # фильтр мусора
        if area < 3000:
            continue

        if mask[click_y, click_x] == 0:
            continue

        if area > best_area:
            best_area = area
            best_mask = mask

    if best_mask is None:
        return False

    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

    cv2.imwrite(save_path, best_mask)
    return True