from ultralytics import YOLO
import torch

model = YOLO('yolov8x')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Active device: ", device)

result = model.predict('Input_video/video_of_match.mp4', save=True, device=device)
print(result[0])
print("========================")
for box in result[0].boxes:
    print(box)
