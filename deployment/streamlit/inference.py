import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

_checkpoint = torch.load("models/best_resnet50.pth", map_location="cpu")

def _build_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.40),
        nn.Linear(256, 1),
    )
    model.load_state_dict(_checkpoint["model_state_dict"])
    model.eval()
    return model

_model = _build_model()

_transform = transforms.Compose([
    transforms.Resize(_checkpoint["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=_checkpoint["normalize_mean"], std=_checkpoint["normalize_std"]),
])

_class_names = _checkpoint["class_names"]
_threshold = _checkpoint["decision_threshold"]

def predict_deepfake(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(image).unsqueeze(0)

    with torch.no_grad():
        logit = _model(tensor)
        fake_prob = torch.sigmoid(logit).item()

    label = _class_names[1] if fake_prob >= _threshold else _class_names[0]
    confidence = fake_prob if label == _class_names[1] else 1 - fake_prob

    return {"label": label, "confidence": confidence}
