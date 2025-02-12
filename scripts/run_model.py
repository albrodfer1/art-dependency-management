import av
import numpy as np
import torch

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

def predict_function(x):
    inputs = image_processor(list(x), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.detach().cpu().numpy()


# video clip consists of 300 frames (10 seconds at 30 FPS)
# file_path = hf_hub_download(
#     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
# )

file_path = "/content/drive/MyDrive/art-video-classification/in/video.mp4"
container = av.open(file_path)

# sample 32 frames
indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container=container, indices=indices)

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400", torchscript= True)
model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400", torchscript= True)

inputs = image_processor(list(video), return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# model predicts one of the 400 Kinetics-400 classes
# predicted_label = logits.argmax(-1).item()
# print(f"predicted_label: {predicted_label}")
# print(model.config.id2label[predicted_label])

logits = outputs[0]  # Extract the logits tensor from the tuple

# Convert logits to predicted class indices by applying argmax
predictions = torch.argmax(logits, dim=1)  # This gives you the index of the highest logit for each sample

# Convert predictions to numpy if needed
predictions = predictions.cpu().numpy()  # Move to CPU and convert to numpy array

print(predictions)  # This will give you the predicted class indices

print(video.shape])

# define classifier

# prepare mean and std arrays for ART classifier preprocessing
# TODO: investigate those values
mean = np.array([0.485, 0.456, 0.406] * (32 * 224 * 224)).reshape((3, 32, 224, 224), order='F')
std = np.array([0.229, 0.224, 0.225] * (32 * 224 * 224)).reshape((3, 32, 224, 224), order='F')

loss_fn = torch.nn.CrossEntropyLoss()
classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=loss_fn,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    input_shape=video.shape[1
    preprocessing=(mean, std),
    nb_classes=400
)

# verify that ART classifier predictions are consistent with original model:
pred = classifier.predict(**inputs)
print(f"predicted_label: {pred}")
