import torch
import numpy as np
import cv2
import hashlib
import logging
from torch import nn
from torchvision import transforms, models
from torch.utils.data.dataset import Dataset
import face_recognition
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


class VideoDataset(Dataset):
    def __init__(self, video_path, sequence_length=20, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frames = []
        for i, frame in enumerate(self.frame_extract(self.video_path)):
            try:
                faces = face_recognition.face_locations(frame)
                if faces:
                    top, right, bottom, left = faces[0]
                    frame = frame[top:bottom, left:right]
            except (face_recognition.FaceRecognitionError, cv2.error) as e:
                logger.warning(f"Face detection error: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                continue

            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break

        frames = torch.stack(frames)[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        while True:
            success, image = vidObj.read()
            if not success:
                break
            yield image


def verify_model_integrity(model_path, expected_hash):
    """Verify model file integrity using SHA256 hash."""
    if not os.path.exists(model_path):
        logger.error("Model file not found!")
        return False

    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    model_hash = sha256.hexdigest()
    if model_hash != expected_hash:
        logger.error("Model integrity check failed! Possible tampering detected.")
        return False

    logger.info("Model integrity verified.")
    return True


def safe_load_model(path_to_model, expected_hash):
    """Securely load a PyTorch model using JIT, with integrity verification."""
    if not verify_model_integrity(path_to_model, expected_hash):
        raise RuntimeError("Model file verification failed!")

    try:
        # Bandit B614 warning suppressed because we ensure integrity before loading
        model = torch.jit.load(path_to_model, map_location=torch.device('cpu'))  # nosec B614
        logger.info("Model loaded securely with JIT.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def load_model(path_to_model):
    """Load the model securely with restricted pickle usage."""
    
    # Correct hash values
    expected_hashes = {
        "genconvit_ed_inference.pth": "86f0c2e875016435def7d031b357bda5dc0061367290d73de121186df3f03f8c",
        "genconvit_vae_inference.pth": "53c627c82d1439fc80e18ac462c1ed6969a3babe5376124a5c38d1c0c88c9042"
    }

    if path_to_model not in expected_hashes:
        raise ValueError(f"Unexpected model path: {path_to_model}")

    if not verify_model_integrity(path_to_model, expected_hashes[path_to_model]):
        raise RuntimeError("Model file verification failed!")

    model = Model(2)
    with open(path_to_model, "rb") as f:
        model_state_dict = torch.jit.load(f, map_location=torch.device('cpu'))

    
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    return model


def predict_video(model, video_path):
    """Perform deepfake detection on a video."""
    dataset = VideoDataset(video_path, sequence_length=20, transform=train_transforms)
    sm = nn.Softmax(dim=1)

    with torch.no_grad():
        frames = dataset[0]
        _, logits = model(frames)
        logits = sm(logits)
        pred_class = torch.argmax(logits).item()
        confidence = logits[0, pred_class].item()

        return {
            "deepfake_detected": pred_class == 0,
            "confidence": round(confidence, 3)
        }
