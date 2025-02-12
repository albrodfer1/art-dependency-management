import av
import numpy as np
import torch
import cv2
from transformers import VivitImageProcessor, VivitForVideoClassification
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

# --- Load Video & Preprocess ---
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))  # Convert to NumPy array
    return np.stack(frames)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len).astype(np.int64)
    return indices

# Load video
file_path = "/content/drive/MyDrive/art-video-classification/in/video.mp4"
container = av.open(file_path)

indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
video_frames = read_video_pyav(container, indices)

# Process frames for the model
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

# Move the model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Process the inputs and move them to the same device
inputs = image_processor(list(video_frames), return_tensors="pt")
x = inputs["pixel_values"].to(device)  # Move to the same device as the model

# --- Define Custom `predict` Method ---
def custom_predict(x):
    # Perform the model forward pass to get logits
    with torch.no_grad():
        outputs = model(input_ids=x)
        logits = outputs.logits
    return logits.cpu().numpy()  # Return logits as NumPy array

# --- Wrap Model with ART PyTorchClassifier ---
loss_fn = torch.nn.CrossEntropyLoss()
classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=loss_fn,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    input_shape=x.shape[1:],
    nb_classes=400,
    predict_function=custom_predict  # Pass the custom predict function
)

# --- Generate Adversarial Example ---
attack = FastGradientMethod(estimator=classifier, eps=0.03)  # Epsilon controls perturbation strength

# Generate adversarial example
x_adv = attack.generate(x=x.cpu().numpy())  # Pass the CPU tensor to ART

# Convert adversarial tensor back to images
x_adv_np = (x_adv * 255).astype(np.uint8)  # Rescale to 0-255
adv_frames = [frame.transpose(1, 2, 0) for frame in x_adv_np[0]]  # Convert (C, H, W) -> (H, W, C)

# --- Save as Video ---
output_path = "/content/drive/MyDrive/art-video-classification/out/adv_video.mp4"
fps = 30  # Adjust based on original video
height, width, _ = adv_frames[0].shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for frame in adv_frames:
    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

video_writer.release()
print(f"Adversarial video saved at: {output_path}")

# --- Classify Adversarial Video ---
inputs_adv = image_processor(list(adv_frames), return_tensors="pt").to(device)
with torch.no_grad():
    outputs_adv = model(**inputs_adv)
    logits_adv = outputs_adv.logits

# Get the predicted class for the adversarial video
predicted_label_adv = logits_adv.argmax(-1).item()

# Print the predicted label for the adversarial video
print(f"Predicted label for the adversarial video: {model.config.id2label[predicted_label_adv]}")
