import cv2
import dlib
import numpy as np

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image
image = cv2.imread("img.png")
img_height, img_width, _ = image.shape

# Convert the image to grayscale for detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)


def extend_points(points, margin=20):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    extended_points = []

    for point in points:
        # Calculate direction vector from centroid to current point
        direction = point - centroid
        # Normalize the direction
        norm = np.linalg.norm(direction)
        if norm == 0:
            norm = 1
        direction_normalized = direction / norm
        # Extend the point away from the centroid
        extended_point = point + direction_normalized * margin
        extended_points.append(extended_point)

    return np.array(extended_points, np.int32)


def extract_feature_with_expanded_points(image, points):
    # Create a full-size mask for the feature
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    # Extract the feature area from the original image using the mask
    feature_extracted = cv2.bitwise_and(image, image, mask=mask)

    # Create an RGBA image from the extracted feature
    feature_rgba = cv2.cvtColor(feature_extracted, cv2.COLOR_BGR2BGRA)
    # Apply the mask as the alpha channel
    feature_rgba[:, :, 3] = mask

    # Use the points to find the bounding rectangle, then crop
    x, y, w, h = cv2.boundingRect(points)
    feature_image = feature_rgba[y:y + h, x:x + w]

    return feature_image


for index, face in enumerate(faces):
    landmarks = predictor(gray, face)

    # Define points arrays for eyes and mouth
    left_eye_points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)])
    right_eye_points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])
    mouth_points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(48, 68)])

    # Extend points
    left_eye_expanded = extend_points(left_eye_points, margin=20)
    right_eye_expanded = extend_points(right_eye_points, margin=20)
    mouth_expanded = extend_points(mouth_points, margin=20)

    # Extract features with expanded points
    left_eye_image = extract_feature_with_expanded_points(image, left_eye_expanded)
    right_eye_image = extract_feature_with_expanded_points(image, right_eye_expanded)
    mouth_image = extract_feature_with_expanded_points(image, mouth_expanded)

    # Save the features as PNG images with transparency
    cv2.imwrite(f"extended_left_eye_{index}.png", left_eye_image)
    cv2.imwrite(f"extended_right_eye_{index}.png", right_eye_image)
    cv2.imwrite(f"extended_mouth_{index}.png", mouth_image)

print("Feature images with extended points and transparent background saved.")
