from enum import Enum

import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from extract_head import extract_head

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task',
                                  delegate=python.BaseOptions.Delegate.CPU)
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=False,
                                       output_facial_transformation_matrixes=False,
                                       num_faces=1,
                                       min_face_detection_confidence = 0.2,
                                       min_face_presence_confidence = 0.2,
                                       min_tracking_confidence = 0.2)
detector = vision.FaceLandmarker.create_from_options(options)
# Initialize MediaPipe Face Mesh


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


def apply_global_homography(orig_image, orig_points_list, pose_src_points_list):
    src_points = np.vstack(orig_points_list)


    dst_points = np.vstack(pose_src_points_list)

    # Compute the homography matrix
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

    # Apply the homography to warp the source image
    warped_src = cv2.warpPerspective(orig_image, H, (orig_image.shape[1], orig_image.shape[0]))
    return warped_src

    orig_points = np.vstack(orig_points_list)
    pose_src_points = np.vstack(pose_src_points_list)

    # Compute the homography matrix
    H, status = cv2.findHomography(orig_points, pose_src_points, cv2.RANSAC)

    # Get image dimensions
    height, width = orig_image.shape[:2]

    # Apply the homography to the corner points of the original image
    corners = np.float32([[0, 0], [0, height], [width, 0], [width, height]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)

    # Determine the bounds of the transformed content
    x_min, y_min = np.min(transformed_corners, axis=0).ravel()
    x_max, y_max = np.max(transformed_corners, axis=0).ravel()

    # Calculate the size needed to fit the transformed content
    new_width = int(np.ceil(x_max - x_min))
    new_height = int(np.ceil(y_max - y_min))

    if (new_width > width) or (new_height > height):
        # new img is big enough to contain the original image
        return cv2.warpPerspective(orig_image, H, (orig_image.shape[1], orig_image.shape[0]))

    # Calculate scaling factors to at least fill the original dimensions
    scale_x = width / new_width
    scale_y = height / new_height
    scale = max(scale_x, scale_y)  # Ensure we fill at least one dimension completely

    # Scaling matrix to apply the determined scale
    scaling_matrix = np.array([[scale, 0, 0],
                               [0, scale, 0],
                               [0, 0, 1]])

    # Adjust the homography matrix to translate and scale the transformed content
    translation_matrix = np.array([[1, 0, -x_min],
                                   [0, 1, -y_min],
                                   [0, 0, 1]])
    adjusted_H = scaling_matrix @ translation_matrix @ H

    # Warp the original image using the adjusted homography matrix
    output_size = (int(scale * new_width), int(scale * new_height))
    warped_image = cv2.warpPerspective(orig_image, adjusted_H, output_size)

    return warped_image

# Function to extract facial landmarks
def get_landmarks(image, get_z=False):
    img_height = image.height
    img_width = image.width

    detection_result = detector.detect(image)
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        if get_z:
            return [(int(lm.x * img_width), int(lm.y * img_height), int(lm.z * 3000)) for lm in landmarks]
        else:
            return [(int(lm.x * img_width), int(lm.y * img_height)) for lm in landmarks]
    print("No landmarks detected in image.")
    return None


class HeadPost(Enum):
    LEFT = "templates/person_looking_forward.jpg"
    RIGHT = "templates/person_looking_forward.jpg"
    DOWN = "templates/person_looking_forward.jpg"
    UP = "templates/person_looking_forward.jpg"
    FORWARD = "templates/person_looking_right.jpg"
def estimate_headposition(img_h, img_w, landmarks):
    face_2d = []
    face_3d = []

    if landmarks:
        for idx, lm in enumerate(landmarks):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm[0], lm[1])
                    nose_3d = (lm[0], lm[1], lm[2])
                x, y = lm[0], lm[1]

                face_2d.append([x, y])
                face_3d.append(([x, y, lm[2]]))

            # Get 2d Coord
        face_2d = np.array(face_2d, dtype=np.float64)

        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w

        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                               [0, focal_length, img_w / 2],
                               [0, 0, 1]])
        distortion_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

        # getting rotational of face
        rmat, jac = cv2.Rodrigues(rotation_vec)

        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        threshold = 4 # def = 10

        # here based on axis rot angle is calculated
        if y < -threshold:
            text = HeadPost.LEFT
        elif y > threshold:
            text = HeadPost.RIGHT
        elif x < -threshold:
            text = HeadPost.DOWN
        elif x > threshold:
            text = HeadPost.UP
        else:
            text = HeadPost.FORWARD

        print("Head looking: ", text)
        return text
    else:
        print("No landmarks detected in image.")
        return None


# Paths to your source and destination (template) images
src_imgage_name='tech'


orig_image_path = f'test/{src_imgage_name}.png'
orig_image = cv2.imread(orig_image_path)


# Extract landmarks from both images
orig_landmarks = get_landmarks(mp.Image(image_format=mp.ImageFormat.SRGB, data=orig_image), get_z=True)

pose_src_image_path = estimate_headposition(orig_image.shape[0], orig_image.shape[1], orig_landmarks)
#orig_landmarks has x, y, z -> remove z
orig_landmarks = [lm[:2] for lm in orig_landmarks]
pose_src_image = cv2.imread(pose_src_image_path.value or "templates/person_looking_forward.jpg")

pose_src_landmarks = get_landmarks(mp.Image(image_format=mp.ImageFormat.SRGB, data=pose_src_image))
# Aggregate source and destination points for all features
orig_points_list = [np.float32([orig_landmarks[i] for i in indices]) for indices in
                    [all_indices]]
pose_src_points_list = [np.float32([pose_src_landmarks[i] for i in indices]) for indices in
                        [all_indices]]

# Apply the global homography based on aggregated points
corrected_image = apply_global_homography(orig_image, orig_points_list, pose_src_points_list)

print("Perspective correction applied.")
extract_dir = f'extracted_{src_imgage_name}'
# Create directory for extracted images
os.makedirs(extract_dir, exist_ok=True)

cv2.imwrite(f'{extract_dir}/corrected.png', corrected_image)
# Re-extract landmarks from the corrected image to get updated positions
corrected_landmarks = get_landmarks(mp.Image(image_format=mp.ImageFormat.SRGB, data=corrected_image))  # Assuming the corrected image is saved and loaded here

# Assuming dst_landmarks are the target landmarks after homography

# Loop through each feature, apply homography, re-extract landmarks, and extract features
for feature, feature_indices, feature_margin_indices in [
    ("left_eye", left_eye_indices, left_eye_margin_indices),
    ("right_eye", right_eye_indices, right_eye_margin_indices),
    ("mouth", mouth_indices, mouth_margin_indices)]:
    # Before extracting features using indices
    if corrected_landmarks:
        corrected_feature_coords = [corrected_landmarks[idx] for idx in feature_indices if idx < len(corrected_landmarks)]
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

detector.close()
print("Facial feature images with perspective correction saved.")
extract_head(f'{extract_dir}/corrected.png')