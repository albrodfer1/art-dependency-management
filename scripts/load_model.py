from transformers import VivitForVideoClassification, AutoImageProcessor
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

# Load the model and processor
model_id = "google/vivit-b-16x2-kinetics400"  # Pretrained on Kinetics-400 dataset
model = VivitForVideoClassification.from_pretrained(model_id)
processor = AutoImageProcessor.from_pretrained(model_id)