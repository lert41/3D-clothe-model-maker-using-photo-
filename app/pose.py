import mediapipe as mp
import cv2
import math

mp_pose = mp.solutions.pose


def estimate_shoulder_width(image_path):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        result = pose.process(rgb)

    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks.landmark

    left = landmarks[11]
    right = landmarks[12]

    h, w, _ = image.shape

    x1, y1 = int(left.x * w), int(left.y * h)
    x2, y2 = int(right.x * w), int(right.y * h)

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
