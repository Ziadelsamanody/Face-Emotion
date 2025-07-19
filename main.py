import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
import numpy as np
from PIL import Image


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])




model = resnet50(pretrained=False)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)
model.load_state_dict(torch.load('fer2013_resnet50.pth'))  # Load trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
               4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


def detect_and_predict_emotion(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))  # Match FER-2013 size
        face_pil = Image.fromarray(face_roi)  # Convert to PIL Image

        # Apply transforms
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Predict emotion
        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = emotion_map[predicted.item()]
                # Draw rectangle and label on image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display or save the result
    cv2.imshow('Emotion Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def detect_and_predict_emotion_video():
    cap = cv2.VideoCapture(0)  # Use webcam (0) or specify video file path
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam or video file.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from video stream.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=3, 
            minSize=(20, 20)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_pil = Image.fromarray(face_roi)
            
            try:
                face_tensor = transform(face_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(face_tensor)
                    _, predicted = torch.max(outputs, 1)
                    emotion = emotion_map[predicted.item()]
            except Exception as e:
                print(f"Error processing face in video at ({x}, {y}, {w}, {h}): {e}")
                continue

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
# 7. Example Usage
if __name__ == "__main__":
    # detect_and_predict_emotion("test.jpg")ضض
    detect_and_predict_emotion_video()
