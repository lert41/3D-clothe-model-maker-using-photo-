import cv2
import numpy as np
import trimesh
from shapely.geometry import Polygon


def get_contour(mask_path):

    mask = cv2.imread(mask_path, 0)

    if mask is None:
        return None

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    epsilon = 0.003 * cv2.arcLength(largest, True)

    approx = cv2.approxPolyDP(
        largest,
        epsilon,
        True
    )

    pts = approx[:, 0, :]

    return pts


def normalize_points(points):

    points = points.astype(np.float32)

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)

    size = max_xy - min_xy

    points -= min_xy

    scale = max(size)

    if scale == 0:
        scale = 1

    points /= scale

    points[:, 1] *= -1

    return points


def generate_3d_model(front_mask, side_mask, output_path):

    front_pts = get_contour(front_mask)

    if front_pts is None:
        print("NO FRONT CONTOUR")
        return None

    front_pts = normalize_points(front_pts)

    polygon = Polygon(front_pts)

    if not polygon.is_valid:
        print("INVALID POLYGON")
        return None

    # depth from side mask
    depth = 0.25

    side_pts = get_contour(side_mask)

    if side_pts is not None:

        x, y, w, h = cv2.boundingRect(side_pts)

        depth = max(w / 400.0, 0.2)

    try:

        mesh = trimesh.creation.extrude_polygon(
            polygon,
            depth
        )

        # поворот
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(
                np.radians(90),
                [1, 0, 0]
            )
        )

        # центрирование
        mesh.apply_translation(
            -mesh.centroid
        )

        mesh.export(output_path)

        print("MODEL SAVED:", output_path)

        return output_path

    except Exception as e:

        print("MESH ERROR:", e)

        return None