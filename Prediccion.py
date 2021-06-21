import cv2
import torch
import torchvision
from torchvision import datasets, transforms

model_nuevo = torch.load('./my_model_45.pt')
model_nuevo.eval()




img1 = cv2.imread('1.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

# Define a transform to normalize the data
elpepe = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5),
                                ])
elpepe_img = elpepe(gray1)
print(elpepe_img.shape)

img = elpepe_img.view(1, 784)

# Turn off gradients to speed up this part
with torch.no_grad():
   logps = model_nuevo(img)

 # Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))


