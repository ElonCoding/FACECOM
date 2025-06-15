import cv2
import torch
import numpy as np
from model import create_model # Assuming create_model can load a pre-trained model

# Configuration (can be loaded from a config file or passed as args)
CONFIG = {
    'img_size': 300,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'checkpoint_path': 'checkpoints/best_model.pth', # Path to your trained model checkpoint
    'num_identities': 2, # Updated based on annotations.json (id1, id2)
}

def load_model(config):
    model = create_model(config, config['num_identities'])
    checkpoint = torch.load(config['checkpoint_path'], map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config['device'])
    model.eval()
    return model

def detect_gender_and_identity_live():
    # Load pre-trained model
    model = load_model(CONFIG)

    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Extract face ROI and convert to RGB
            face_roi = frame[y:y+h, x:x+w]
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Preprocess face ROI for model input
            face_roi_resized = cv2.resize(face_roi_rgb, (CONFIG['img_size'], CONFIG['img_size']))
            face_roi_tensor = torch.from_numpy(face_roi_resized).permute(2, 0, 1).float() / 255.0
            # Normalize (ImageNet mean/std, adjust if your model uses different)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            face_roi_tensor = (face_roi_tensor - mean) / std

            face_roi_tensor = face_roi_tensor.unsqueeze(0).to(CONFIG['device'])

            # Perform inference
            with torch.no_grad():
                outputs = model(face_roi_tensor)
                gender_output = outputs['gender']
                identity_output = outputs['identity']

            gender_pred = (torch.sigmoid(gender_output).cpu().item() > 0.5) # Assuming binary gender classification
            identity_pred = torch.argmax(identity_output, dim=1).cpu().item()

            gender_label = "Female" if gender_pred else "Male"
            identity_label = f"ID: {identity_pred}"

            # Display gender and identity labels
            cv2.putText(frame, gender_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.putText(frame, identity_label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Live Gender and Identity Detection', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_gender_and_identity_live()