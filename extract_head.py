import cv2
import numpy as np
import mediapipe as mp

FACEMESH_LIPS = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)]

FACEMESH_LEFT_EYE = [(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)]

FACEMESH_LEFT_IRIS = [(474, 475), (475, 476), (476, 477),
                                (477, 474)]

FACEMESH_LEFT_EYEBROW = [(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)]

FACEMESH_RIGHT_EYE = [(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)]

FACEMESH_RIGHT_EYEBROW = [(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)]

FACEMESH_RIGHT_IRIS = [(469, 470), (470, 471), (471, 472),
                                 (472, 469)]

FACEMESH_FACE_OVAL = [(10, 338), (338, 297), (297, 332), (332, 284),
                                (284, 251), (251, 389), (389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162), (162, 21), (21, 54),
                                (54, 103), (103, 67), (67, 109), (109, 10)]

FACEMESH_NOSE = [(168, 6), (6, 197), (197, 195), (195, 5),
                           (5, 4), (4, 1), (1, 19), (19, 94), (94, 2), (98, 97),
                           (97, 2), (2, 326), (326, 327), (327, 294),
                           (294, 278), (278, 344), (344, 440), (440, 275),
                           (275, 4), (4, 45), (45, 220), (220, 115), (115, 48),
                           (48, 64), (64, 98)]

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)

def extract_head(image_path):
    image = cv2.imread(image_path)

    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect the face and facial landmarks
    results = face_mesh.process(image_rgb)

    # Check if at least one face is detected
    if results.multi_face_landmarks:
        # Create an empty mask to draw the facial landmarks
        mask = np.zeros((height, width), dtype=np.uint8)

        for face_landmarks in results.multi_face_landmarks:
            # Initialize a list to store the points of the face oval
            points = []

            # Iterate through each connection in the FACEMESH_FACE_OVAL
            for start_idx, end_idx in FACEMESH_FACE_OVAL:
                # Add the start point of each connection to the list
                start_point = face_landmarks.landmark[start_idx]
                points.append((int(start_point.x * width), int(start_point.y * height)))

                # Optionally, you can add the end point as well, but it's not necessary
                # since each point will be included as a start point in subsequent connections

            # Convert points to a numpy array
            points = np.array(points, dtype=np.int32)

            # Draw the face oval on the mask
            cv2.fillPoly(mask, [points], 255)

            # Calculate the bounding box of the face oval
            x, y, w, h = cv2.boundingRect(np.array([points]))

            extend_up = h // 3  # Extend upwards for the top of the head
            extend_down = h // 3  # Extend downwards to include the chin
            extend_sides = w * 10 // 100  # Extend sides by 10% of the width

            # Adjust the bounding box
            y, h = y - extend_up, h + extend_up + extend_down
            x, w = x - extend_sides, w + 2 * extend_sides

            # Ensure the adjustments do not exceed image boundaries
            x, y = max(x, 0), max(y, 0)
            w, h = min(width - x, w), min(height - y, h)

            # Draw the adjusted bounding box on the mask to include more of the sides
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            # Find the bounding box of the head based on the mask
            x, y, w, h = cv2.boundingRect(mask)

            # Create the RGBA image based on the bounding box
            cropped_rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
            extracted_portion = cv2.bitwise_and(image[y:y + h, x:x + w], image[y:y + h, x:x + w],
                                                mask=mask[y:y + h, x:x + w])
            alpha_channel = mask[y:y + h, x:x + w]
            cropped_rgba_image[..., :3] = extracted_portion
            cropped_rgba_image[..., 3] = alpha_channel

            # Save or display the cropped RGBA image
            output_path = image_path.replace(".png", "_extracted.png")
            cv2.imwrite(output_path, cropped_rgba_image)
            print(f"Extracted and cropped image saved to: {output_path}")

    else:
        print("No face detected.")

    # Release resources
    face_mesh.close()


def auto_crop_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define thresholds for what is considered 'black' or 'white'
    black_threshold = 15  # Pixel values below this are considered black
    white_threshold = 240  # Pixel values above this are considered white

    # Initialize indices for cropping
    top, bottom = 0, gray.shape[0] - 1
    left, right = gray.shape[1] - 1, 0

    # Function to determine the background color at the edges
    def determine_background_color():
        # Check corners of the image [top-left, top-right, bottom-left, bottom-right]
        corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
        white_count = sum([1 for corner in corners if corner > white_threshold])
        black_count = sum([1 for corner in corners if corner < black_threshold])

        if white_count > black_count:
            return 'white'
        else:
            return 'black'

    background_color = determine_background_color()

    # Adjust checks based on the detected background color
    if background_color == 'white':
        check_condition = lambda x: x > white_threshold
    else:
        check_condition = lambda x: x < black_threshold

    # Check top and bottom
    while top < bottom and np.all(check_condition(gray[top])):
        top += 1
    while bottom > top and np.all(check_condition(gray[bottom])):
        bottom -= 1

    # Check left and right
    while left > right and np.all(check_condition(gray[:, left])):
        left -= 1
    while right < left and np.all(check_condition(gray[:, right])):
        right += 1

    # Ensure there is still an image to crop
    if top >= bottom or left <= right:
        return image  # Return the original image if the cropping criteria are too extensive

    # Crop the image based on the found indices
    cropped_image = image[top:bottom + 1, right:left + 1]

    return cropped_image

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(static_image_mode=True, model_complexity=2, enable_segmentation=True)
def extract_person(image_path):
    output_path = image_path.replace(".png", "")

    # Load the image
    image = cv2.imread(image_path)

    # Check if image is loaded
    if image is None:
        print(f"Error: Image not found at {image_path}")
        return

    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = holistic.process(image)

    if results.pose_landmarks:
        # Draw pose landmarks
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Convert back to BGR for saving and displaying
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Save the annotated image
        cv2.imwrite(f"{output_path}_annotated.png", annotated_image)
        print(f"Annotated image saved to {output_path}_annotated.png")

        if results.segmentation_mask is not None:
            # Apply a binary threshold to create a binary mask
            segmentation_mask = results.segmentation_mask > 0.6
            segmentation_mask = segmentation_mask.astype(np.uint8) * 255

            # Calculate the bounding rectangle of the segmentation mask
            x, y, w, h = cv2.boundingRect(segmentation_mask)

            # Extract the rectangular area from the original image
            extracted_rectangle = image[y:y + h, x:x + w]
            extracted_person = cv2.cvtColor(extracted_rectangle, cv2.COLOR_RGB2BGR)

            # Save the image with the person extracted
            cv2.imwrite(f'{output_path}_extracted.png', auto_crop_image(extracted_person))
            print(f"Extracted person with rectangular mask saved to {output_path}_extracted.png")
    else:
        print("No pose landmarks detected.")
        extract_head(image_path)