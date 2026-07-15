"""
Face Recognition Project using OpenCV
=======================================
Task: Face Recognition (from Task Instructions - Task 1, OpenCV project)

This script detects human faces using OpenCV's Haar Cascade Classifier.
It supports two modes:
    1. Image mode    -> detect faces in a static image (images/Face.jpg)
    2. Webcam mode   -> detect faces in real time from the camera

Author: (your name here)
"""

import cv2
import os
import sys

# ---------------------------------------------------------
# 1. Paths configuration
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
IMAGE_PATH = os.path.join(BASE_DIR, "images", "Face.jpg")
OUTPUT_PATH = os.path.join(BASE_DIR, "images", "Face_detected.jpg")


# ---------------------------------------------------------
# 2. Load the Haar Cascade classifier
# ---------------------------------------------------------
def load_face_cascade():
    """Load the pretrained Haar Cascade model for frontal face detection."""
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if face_cascade.empty():
        print("Error: could not load haarcascade_frontalface_default.xml")
        sys.exit(1)

    return face_cascade


# ---------------------------------------------------------
# 3. Detect faces in a single image
# ---------------------------------------------------------
def detect_faces_in_image():
    face_cascade = load_face_cascade()

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: could not read image at {IMAGE_PATH}")
        sys.exit(1)

    # Haar cascades work on grayscale images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,     # how much the image size is reduced at each scale
        minNeighbors=5,      # how many neighbors each candidate rectangle should have
        minSize=(40, 40)     # minimum possible face size
    )

    print(f"Faces detected: {len(faces)}")

    # Draw a rectangle around every detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img, "Face", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    # Save the result
    cv2.imwrite(OUTPUT_PATH, img)
    print(f"Result saved to: {OUTPUT_PATH}")

    # Show the result
    cv2.imshow("Face Detection - Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------
# 4. Detect faces in real time using the webcam
# ---------------------------------------------------------
def detect_faces_webcam():
    face_cascade = load_face_cascade()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not access the webcam.")
        sys.exit(1)

    print("Press 'q' to quit the webcam window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to grab frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, "Face", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        cv2.imshow("Face Detection - Webcam (press 'q' to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Face Recognition Project - OpenCV")
    print("1. Detect faces in the sample image (images/Face.jpg)")
    print("2. Detect faces live using the webcam")

    choice = input("Choose an option (1/2): ").strip()

    if choice == "1":
        detect_faces_in_image()
    elif choice == "2":
        detect_faces_webcam()
    else:
        print("Invalid choice. Please run again and choose 1 or 2.")
