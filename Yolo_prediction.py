from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Active device: ", device)

model = YOLO("yolov8x.pt")

result = model.predict('Input_video/08fd33_4.mp4', save=True, device=device)
print(result[0])
print("========================")
for box in result[0].boxes:
    print(box)
