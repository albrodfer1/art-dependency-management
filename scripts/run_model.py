import av
import numpy as np
import torch
import cv2

from transformers import VivitImageProcessor, VivitForVideoClassification
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
# from huggingface_hub import hf_hub_download

np.random.seed(0)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): Li# from huggingface_hub import hf_hub_download
st of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
# file_path = hf_hub_download(
#     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
# )

file_path = "/content/drive/MyDrive/art-video-classification/in/video.mp4"
container = av.open(file_path)

# sample 32 frames
indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container=container, indices=indices)

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

inputs = image_processor(list(video), return_tensors="pt")
x = inputs["pixel_values"]

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# model predicts one of the 400 Kinetics-400 classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])



# --- Wrap Model with ART PyTorchClassifier ---
loss_fn = torch.nn.CrossEntropyLoss()
classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=loss_fn,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    input_shape=x.shape[1:],
    nb_classes=400,
)

# --- Generate Adversarial Example ---
attack = FastGradientMethod(estimator=classifier, eps=0.03)  # Epsilon controls perturbation strength
x_adv = attack.generate(x=x.numpy())


# --- Predict on Adversarial Example ---
x_adv_tensor = torch.tensor(x_adv)
with torch.no_grad():
    adv_outputs = model(pixel_values=x_adv_tensor)
    adv_logits = adv_outputs.logits

adv_predicted_label = adv_logits.argmax(-1).item()
print("Adversarial Prediction:", model.config.id2label[adv_predicted_label])

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
