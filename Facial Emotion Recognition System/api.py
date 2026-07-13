from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os
import sys

# Import model architecture
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
from models.model import FacialExpressionCNN

app = FastAPI(title="Facial Expression Recognition API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
model = FacialExpressionCNN(num_classes=len(classes)).to(device)
model_path = os.path.join(PROJECT_ROOT, 'outputs', 'best_model.pth')

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
else:
    print(f"Warning: Model not found at {model_path}")

# Image Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.get("/")
async def root():
    return {"status": "online", "model": "FacialExpressionCNN", "classes": classes}

@app.post("/analyze")
async def analyze_expression(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

            prob_list = probs.cpu().numpy()[0].tolist()
            label = classes[pred.item()]

        return {
            "prediction": label,
            "confidence": round(float(conf), 4),
            "probabilities": {classes[i]: round(float(prob_list[i]), 4) for i in range(len(classes))}
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
