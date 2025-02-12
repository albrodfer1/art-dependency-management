import torch
from transformers import AutoModelForVideoClassification, AutoFeatureExtractor
import numpy as np
import cv2
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

# Load the pretrained face emotion recognition model
model_name = "ElenaRyumina/face_emotion_recognition"
model = AutoModelForVideoClassification.from_pretrained(model_name)
extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Ensure model is in evaluation mode
model.eval()

# Define preprocessing function
def preprocess_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in np.linspace(0, total_frames - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    inputs = extractor(frames, return_tensors="pt")
    return inputs["pixel_values"]

# Load and preprocess video
video_path = "/content/drive/MyDrive/art-video-classification/in/video.mp4"  # Replace with an actual video path
video_input = preprocess_video(video_path)

# Define ART classifier
classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=torch.nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    input_shape=video_input.shape,
    nb_classes=model.config.num_labels,
)

# Generate adversarial example
attack = FastGradientMethod(estimator=classifier, eps=0.01)
adversarial_video = attack.generate(x=video_input.numpy())

# Predict on adversarial example
adv_prediction = model(torch.tensor(adversarial_video))
print("Adversarial Prediction:", torch.argmax(adv_prediction.logits, dim=1))
