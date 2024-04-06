import numpy as np
import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)


image = cv2.imread("test/linus.png")

height, width, _ = image.shape
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect the face and facial landmarks
results = face_mesh.process(image_rgb)

img_h, img_w, img_c = image.shape
face_2d = []
face_3d = []

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                x, y = int(lm.x * img_w), int(lm.y * img_h)

                face_2d.append([x, y])
                face_3d.append(([x, y, lm.z]))

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

        threshold = 10

        # here based on axis rot angle is calculated
        if y < -threshold:
            text = "Looking Left"
        elif y > threshold:
            text = "Looking Right"
        elif x < -threshold:
            text = "Looking Down"
        elif x > threshold:
            text = "Looking Up"
        else:
            text = "Forward"

        print("text: ", text)