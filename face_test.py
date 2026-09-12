import cv2
import mediapipe as mp
import math

# =========================
# START CAMERA
# =========================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open camera")
    exit()

# =========================
# MEDIAPIPE FACE MESH
# =========================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# =========================
# EYE LANDMARKS
# =========================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# =========================
# VARIABLES
# =========================

blink_count = 0
eye_closed = False


# =========================
# DISTANCE FUNCTION
# =========================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# =========================
# EAR CALCULATION
# =========================

def calculate_ear(landmarks, eye):

    vertical1 = distance(
        landmarks[eye[1]],
        landmarks[eye[5]]
    )

    vertical2 = distance(
        landmarks[eye[2]],
        landmarks[eye[4]]
    )

    horizontal = distance(
        landmarks[eye[0]],
        landmarks[eye[3]]
    )

    if horizontal == 0:
        return 0

    ear = (vertical1 + vertical2) / (2 * horizontal)

    return ear


# =========================
# MAIN LOOP
# =========================

while True:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read camera")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Face detection
    results = face_mesh.process(rgb_frame)

    # =========================
    # FACE FOUND
    # =========================

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        # Calculate both eyes
        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE
        )

        # Average EAR
        ear = (left_ear + right_ear) / 2

        # =========================
        # BLINK DETECTION
        # =========================

        if ear < 0.23:

            if not eye_closed:
                eye_closed = True

        else:

            if eye_closed:
                blink_count += 1
                eye_closed = False

        # =========================
        # FACE DETECTED TEXT
        # =========================

        cv2.putText(
            frame,
            "FACE DETECTED",
            (30, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # =========================
        # BLINK COUNT
        # =========================

        cv2.putText(
            frame,
            f"BLINKS: {blink_count}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        # =========================
        # EAR VALUE
        # =========================

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # =========================
    # FACE NOT FOUND
    # =========================

    else:

        cv2.putText(
            frame,
            "FACE NOT DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"BLINKS: {blink_count}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    # =========================
    # SHOW CAMERA
    # =========================

    cv2.imshow(
        "BlinkAI - Blink Counter",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLOSE EVERYTHING
# =========================

camera.release()

cv2.destroyAllWindows()

print("BlinkAI stopped.")
print(f"Total blinks: {blink_count}")