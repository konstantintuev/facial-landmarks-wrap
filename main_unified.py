import cv2
import dlib
import numpy as np

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image
image_path = "img.png"  # Update the path to your image
image = cv2.imread(image_path)
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
        direction = point - centroid
        norm = np.linalg.norm(direction)
        norm = norm if norm else 1
        direction_normalized = direction / norm
        extended_point = point + direction_normalized * margin
        extended_points.append(extended_point)

    return np.array(extended_points, np.int32)


def extract_feature_with_expanded_points(image, points):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    feature_extracted = cv2.bitwise_and(image, image, mask=mask)
    feature_rgba = cv2.cvtColor(feature_extracted, cv2.COLOR_BGR2BGRA)
    feature_rgba[:, :, 3] = mask

    x, y, w, h = cv2.boundingRect(points)
    feature_image = feature_rgba[y:y + h, x:x + w]

    return feature_image, (x, y, w, h)


# Create a transparent canvas
combined_features = np.zeros((img_height, img_width, 4), dtype=np.uint8)

for index, face in enumerate(faces):
    landmarks = predictor(gray, face)

    # Define and extend points arrays for eyes and mouth
    features = [(np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)]), 'left_eye'),
                (np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)]), 'right_eye'),
                (np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(48, 68)]), 'mouth')]

    for feature_points, feature_name in features:
        extended_points = extend_points(feature_points, margin=20)
        feature_image, (x, y, w, h) = extract_feature_with_expanded_points(image, extended_points)

        # Place each feature back in its original position
        combined_features[y:y + h, x:x + w, :] = feature_image

        # Save individual features as well
        cv2.imwrite(f"{feature_name}_{index}.png", feature_image)

# Save the combined features image
cv2.imwrite("combined_features.png", combined_features)

print("Combined features image saved.")
