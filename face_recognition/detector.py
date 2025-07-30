import face_recognition
import cv2
import numpy as np
from pathlib import Path
import pickle
from PIL import Image, ImageDraw

# Path to store face encodings
ENCODINGS_PATH = Path("encodings.pkl")

def encode_known_faces(model="hog"):
    names = []
    encodings = []
    
    # Iterate through training images
    for filepath in Path("train").glob("*/*.jpg"):
        name = filepath.parent.name  # Folder name is the label
        image = face_recognition.load_image_file(filepath)
        
        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)
    
    # Save encodings to a file
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"names": names, "encodings": encodings}, f)
    print("Face encodings saved to encodings.pkl")

def recognize_faces(image, loaded_encodings, model="hog"):
    # Convert image to RGB (face_recognition expects RGB)
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)
    
    # Convert to BGR for OpenCV display
    output_image = image.copy()
    
    # Recognize faces
    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(output_image, bounding_box, name)
    
    return output_image

def _recognize_face(unknown_encoding, loaded_encodings):
    # Compare unknown encoding with known encodings
    matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding, tolerance=0.6
    )
    for match, name in zip(matches, loaded_encodings["names"]):
        if match:
            return name
    return None

def _display_face(image, bounding_box, name):
    top, right, bottom, left = bounding_box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

def validate(model="hog"):
    # Load known encodings
    with open(ENCODINGS_PATH, "rb") as f:
        loaded_encodings = pickle.load(f)
    
    # Test on validation images
    for filepath in Path("test").rglob("*.jpg"):
        if filepath.is_file():
            print(f"Processing {filepath}")
            image = cv2.imread(str(filepath))
            output_image = recognize_faces(image, loaded_encodings, model=model)
            cv2.imshow("Validation", output_image)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

def real_time_recognition(model="hog"):
    # Load known encodings
    with open(ENCODINGS_PATH, "rb") as f:
        loaded_encodings = pickle.load(f)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Recognize faces in the frame
        output_frame = recognize_faces(frame, loaded_encodings, model=model)
        
        # Display the frame
        cv2.imshow("Real-Time Face Recognition", output_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Step 1: Encode faces from training dataset
    encode_known_faces()
    
    # Step 2: Validate on test images
    validate()
    
    # Step 3: Start real-time recognition
    print("Starting real-time face recognition. Press 'q' to quit.")
    real_time_recognition()