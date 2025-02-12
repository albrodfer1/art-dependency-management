def load_video(video_path, num_frames=16, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in np.linspace(0, frame_count - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(frame)

    cap.release()
    
    frames = np.stack(frames)  # Shape: (num_frames, height, width, 3)
    return frames

video_path = "/content/drive/MyDrive/art-video-classification/in/video.mp4"  # Change to your video file
video_frames = load_video(video_path)

# Normalize the frames
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

video_tensor = torch.stack([transform(frame) for frame in video_frames])  # Shape: (num_frames, 3, 224, 224)
video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension (1, num_frames, 3, 224, 224)
