import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)


def get_landmarks(image_path):
    """Extract facial landmarks using MediaPipe."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        img_h, img_w = image.shape[:2]
        return [(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks]
    return []


# Define your landmark indices arrays
left_eye_indices = [226, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25]
left_eye_margin_indices = [226, 113, 225, 224, 223, 222, 221, 189, 244, 233, 232, 231, 230, 229, 228, 31]
right_eye_indices = [463, 414, 286, 258, 257, 259, 260, 467, 446, 255, 339, 254, 253, 252, 256, 341]
right_eye_margin_indices = [464, 413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453]
mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
mouth_margin_indices = [57, 186, 92, 165, 167, 164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43]

# Paths to your images
src_image_path = 'test/linus.png'
dst_image_path = 'templates/forward.png'  # This could be a frontal face template

# Extract landmarks from both images
src_landmarks = get_landmarks(src_image_path)
dst_landmarks = get_landmarks(dst_image_path)


def apply_local_homography(src_image, src_points, dst_points):
    """Apply homography to a localized feature area and blend it back into the source image."""
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

    # Warp the entire source image to get the transformed feature
    warped_src = cv2.warpPerspective(src_image, H, (src_image.shape[1], src_image.shape[0]))

    # Create a mask from the destination points to isolate the feature
    mask = np.zeros(src_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(dst_points)], 255)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert mask to 3 channel

    # Blend the warped source feature into the original source image
    blended = np.where(mask == 255, warped_src, src_image)
    return warped_src


# Load the source image to apply transformations
src_image = cv2.imread(src_image_path)

# For each feature, compute local homography and blend the feature into the source image
for feature_indices in [left_eye_indices, right_eye_indices, mouth_indices]:
    src_feature_points = np.float32([src_landmarks[i] for i in feature_indices])
    dst_feature_points = np.float32([dst_landmarks[i] for i in feature_indices])

    # Apply local homography to the feature area
    src_image = apply_local_homography(src_image, src_feature_points, dst_feature_points)

# Save the final image
cv2.imwrite('final_image.jpg', src_image)
