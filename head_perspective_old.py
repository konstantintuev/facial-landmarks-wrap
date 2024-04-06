import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# Helper function to convert normalized landmarks to pixel coordinates
def landmarks_to_px(landmarks, width, height):
    return [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]


# Function to create a mask for landmarks
def create_mask(image, landmarks):
    # Ensure landmarks are in a proper format
    landmarks_int = np.array(landmarks, dtype=np.int32).reshape((-1, 1, 2))

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [landmarks_int], 255)  # Use landmarks_int here

    return mask


# Function to apply a mask and create a transparent image
def apply_mask(image, mask):
    transparent_img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    transparent_img[:, :, 3] = mask
    return transparent_img


# Define landmark indices for the eyes and their margins
left_eye_indices = [226, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25]
left_eye_margin_indices = [226, 113, 225, 224, 223, 222, 221, 189, 244, 233, 232, 231, 230, 229, 228, 31]
right_eye_indices = [463, 414, 286, 258, 257, 259, 260, 467, 446, 255, 339, 254, 253, 252, 256, 341]
right_eye_margin_indices = [464, 413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453]
mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
mouth_margin_indices = [57, 186, 92, 165, 167, 164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43]

all_indices = list(range(468))  # All landmark indices


# Function to apply local homography across multiple features
def apply_global_homography(src_image, src_points_list, dst_points_list):
    """Apply a single homography transformation based on multiple feature points."""
    # Flatten the list of points
    src_points = np.vstack(src_points_list)
    dst_points = np.vstack(dst_points_list)

    # Compute the homography matrix
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

    # Apply the homography to warp the source image
    warped_src = cv2.warpPerspective(src_image, H, (src_image.shape[1], src_image.shape[0]))
    return warped_src


# Function to extract facial landmarks
def get_landmarks(image):
    img_height, img_width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        return [(int(lm.x * img_width), int(lm.y * img_height)) for lm in landmarks]
    print("No landmarks detected in image.")
    return None


# Paths to your source and destination (template) images
src_image_path = 'test/linus.png'
dst_image_path = 'templates/person_looking_forward.jpg'
src_image = cv2.imread(src_image_path)
dst_image = cv2.imread(dst_image_path)

# Extract landmarks from both images
src_landmarks = get_landmarks(src_image)
dst_landmarks = get_landmarks(dst_image)
# Aggregate source and destination points for all features
src_points_list = [np.float32([src_landmarks[i] for i in indices]) for indices in
                   [all_indices]]
dst_points_list = [np.float32([dst_landmarks[i] for i in indices]) for indices in
                   [all_indices]]

# Apply the global homography based on aggregated points
corrected_image = apply_global_homography(src_image, src_points_list, dst_points_list)

# Re-extract landmarks from the corrected image to get updated positions
corrected_landmarks = get_landmarks(corrected_image)  # Assuming the corrected image is saved and loaded here

print("Perspective correction applied.")
extract_dir = 'extracted_linus_old'
# Create directory for extracted images
os.makedirs(extract_dir, exist_ok=True)
cv2.imwrite(f'{extract_dir}/corrected.png', corrected_image)

# Assuming dst_landmarks are the target landmarks after homography

# Loop through each feature, apply homography, re-extract landmarks, and extract features
for feature, feature_indices, feature_margin_indices in [
    ("left_eye", left_eye_indices, left_eye_margin_indices),
    ("right_eye", right_eye_indices, right_eye_margin_indices),
    ("mouth", mouth_indices, mouth_margin_indices)]:
    # Before extracting features using indices
    if corrected_landmarks:
        corrected_feature_coords = [corrected_landmarks[idx] for idx in feature_indices if
                                    idx < len(corrected_landmarks)]
        corrected_feature_margin_coords = [corrected_landmarks[idx] for idx in feature_margin_indices if
                                           idx < len(corrected_landmarks)]
    else:
        # Handle the case where no landmarks were detected in the corrected image
        continue  # Or other appropriate handling

    feature_mask = create_mask(corrected_image, corrected_feature_coords)
    feature_margin_mask = create_mask(corrected_image, corrected_feature_margin_coords)
    only_margin_mask = cv2.subtract(feature_margin_mask, feature_mask)

    x, y, w, h = cv2.boundingRect(np.array(corrected_feature_margin_coords))
    feature_with_margin_image = apply_mask(corrected_image, feature_margin_mask)[y:y + h, x:x + w]
    feature_margin_only_image = apply_mask(corrected_image, only_margin_mask)[y:y + h, x:x + w]
    feature_only_image = apply_mask(corrected_image, feature_mask)[y:y + h, x:x + w]

    # Save the images
    cv2.imwrite(f'{extract_dir}/{feature}_with_margin_corrected.png', feature_with_margin_image)
    cv2.imwrite(f'{extract_dir}/{feature}_margin_only_corrected.png', feature_margin_only_image)
    cv2.imwrite(f'{extract_dir}/{feature}_only_corrected.png', feature_only_image)

face_mesh.close()
print("Facial feature images with perspective correction saved.")