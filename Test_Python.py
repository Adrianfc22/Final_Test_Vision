from PIL import Image
import cv2
import torch
import torchvision
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
print(device)
mod = torch.load('my_model_35.pt')

transform = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

img = Image.open('2no.jpg')
img = transform(img)
img = img.unsqueeze(0)

x = img.to(device)
out = mod(x)
_, pred = torch.max(out.data, 1)
print(pred)

if pred[0] == 0:
  print("Normal")
else:
  print("Tumor")
