# This script was moved to the face_mesh_project directory.

import os
import sys
import io
import cv2
import mediapipe as mp
import numpy as np
import contextlib

# Buffer for debug output
_debug_buffer = io.StringIO()
_stdout = sys.stdout
_stderr = sys.stderr
sys.stdout = _debug_buffer
sys.stderr = _debug_buffer

# MediaPipe Face Mesh initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Includes iris/eye landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing setup
mp_drawing = mp.solutions.drawing_utils
dot_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Landmark indexes
FACE_OUTLINE_IDXS = list(range(0, 17))         # Jawline
LEFT_EYE_IDXS = list(range(474, 478))          # Left iris area
RIGHT_EYE_IDXS = list(range(469, 473))         # Right iris area

# EAR landmarks for eye closure detection
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
EAR_THRESHOLD = 0.2

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    sys.stdout = _stdout
    sys.stderr = _stderr
    print(_debug_buffer.getvalue())
    exit()

# Function to estimate head direction (left/right/forward)
def get_head_direction(landmarks, image_w, image_h):
    left_cheek_x = landmarks[234].x * image_w
    right_cheek_x = landmarks[454].x * image_w
    nose_x = landmarks[1].x * image_w

    face_center_x = (left_cheek_x + right_cheek_x) / 2
    face_width = right_cheek_x - left_cheek_x
    threshold = face_width * 0.1  # 10% of face width

    if nose_x < face_center_x - threshold:
        return "Turned Left"
    elif nose_x > face_center_x + threshold:
        return "Turned Right"
    else:
        return "Forward"

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, eye_indices, image_w, image_h):
    coords = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_indices]
    vertical_1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    vertical_2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    horizontal = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw jawline
            for idx in FACE_OUTLINE_IDXS:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

            # Draw iris/eye points
            for idx in LEFT_EYE_IDXS + RIGHT_EYE_IDXS:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            # Head direction
            direction = get_head_direction(face_landmarks.landmark, w, h)

            # EAR calculation for both eyes
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE_LANDMARKS, w, h)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE_LANDMARKS, w, h)

            # Determine eye state
            if left_ear < EAR_THRESHOLD and right_ear >= EAR_THRESHOLD:
                eye_state = "Right Eye Closed"
            elif right_ear < EAR_THRESHOLD and left_ear >= EAR_THRESHOLD:
                eye_state = "Left Eye Closed"
            elif left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                eye_state = "Both Eyes Closed"
            else:
                eye_state = "Eyes Open"

            # --- Display Info ---

            # Head direction color
            direction_color = (0, 255, 0)  # Green = Forward
            if direction == "Turned Left":
                direction_color = (0, 0, 255)  # Red
            elif direction == "Turned Right":
                direction_color = (255, 0, 0)  # Blue

            # Eye state color
            eye_color = (0, 255, 255)  # Yellow = Open
            if eye_state == "Both Eyes Closed":
                eye_color = (0, 0, 255)
            elif eye_state == "Left Eye Closed":
                eye_color = (0, 128, 255)
            elif eye_state == "Right Eye Closed":
                eye_color = (255, 128, 0)

            # Show text
            cv2.putText(frame, f"Head: {direction}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, direction_color, 2)

            cv2.putText(frame, f"Eyes: {eye_state}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, eye_color, 2)
    else:
        print("No face landmarks detected.")

    # Show output
    cv2.imshow("Face, Eyes, and Head Direction", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
sys.stdout = _stdout
sys.stderr = _stderr
print(_debug_buffer.getvalue())
