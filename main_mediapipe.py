import os

import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Read the image
image_path = 'img.png'  # Replace with your image path
image = cv2.imread(image_path)
img_height, img_width, _ = image.shape
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect the face and landmarks
results = face_mesh.process(image_rgb)

# Define landmark indices for the eyes and their margins
left_eye_indices = [226, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25]
left_eye_margin_indices = [226, 113, 225, 224, 223, 222, 221, 189, 244, 233, 232, 231, 230, 229, 228, 31]
right_eye_indices = [463, 414, 286, 258, 257, 259, 260, 467, 446, 255, 339, 254, 253, 252, 256, 341]
right_eye_margin_indices = [464, 413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453]
mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
mouth_margin_indices = [57, 186, 92, 165, 167, 164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43]

# Helper function to convert normalized landmarks to pixel coordinates
def landmarks_to_px(landmarks, width, height):
    return [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]


# Function to create a mask for landmarks
def create_mask(image, landmarks):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(landmarks)], 255)
    return mask


# Function to apply a mask and create a transparent image
def apply_mask(image, mask):
    # Create an RGBA version of the image
    transparent_img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    # Set the alpha channel to the mask
    transparent_img[:, :, 3] = mask
    return transparent_img

os.makedirs('extracted', exist_ok=True)

if results.multi_face_landmarks:
    # Get the detected facial landmarks.
    face_landmarks = results.multi_face_landmarks[0].landmark

    # Convert landmarks to pixel coordinates
    facial_landmarks = landmarks_to_px(face_landmarks, img_width, img_height)

    # Extract the coordinates for each feature and corresponding margins
    for feature, feature_indices, feature_margin_indices in [
        ("left_eye", left_eye_indices, left_eye_margin_indices),
        ("right_eye", right_eye_indices, right_eye_margin_indices),
        ("mouth", mouth_indices, mouth_margin_indices)
    ]:
        feature_coords = [facial_landmarks[idx] for idx in feature_indices]
        feature_margin_coords = [facial_landmarks[idx] for idx in feature_margin_indices]

        # Create masks for the eye and the margin
        feature_mask = create_mask(image, feature_coords)
        feature_margin_mask = create_mask(image, feature_margin_coords)

        # Subtract the eye mask from the margin mask to get just the margin
        only_margin_mask = cv2.subtract(feature_margin_mask, feature_mask)

        # Find bounding rectangle for the eye with margin to standardize the size of all images
        x, y, w, h = cv2.boundingRect(np.array(feature_margin_coords))

        # Apply masks and crop to the bounding box to get images with transparent backgrounds
        feature_with_margin_image = apply_mask(image, feature_margin_mask)[y:y + h, x:x + w]
        feature_margin_only_image = apply_mask(image, only_margin_mask)[y:y + h, x:x + w]
        feature_only_image = apply_mask(image, feature_mask)[y:y + h, x:x + w]

        # Save the images
        cv2.imwrite(f'extracted/{feature}_with_margin.png', feature_with_margin_image)
        cv2.imwrite(f'extracted/{feature}_margin_only.png', feature_margin_only_image)
        cv2.imwrite(f'extracted/{feature}_only.png', feature_only_image)

face_mesh.close()
print("Eye images saved as PNG.")
