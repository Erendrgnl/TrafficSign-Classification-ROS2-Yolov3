from torchvision import transforms
import torch
import argparse
from PIL import Image
import time
import numpy as np

#Models
from models.vgg import vgg
from models.efficentNet import efficientnet_b0
from models.shuffleNetv2 import shufflenet_v2_x1_0

def main():
    class_names = ["go","stop"]

    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str,help="Image file Path")

    args = parser.parse_args()
    img = Image.open(args.img)
    
    transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.ToTensor()])
    img = transform(img)
    img = img.view((1,3,224,224))

    model = efficientnet_b0(num_classes=2).to(device)
    #model = vgg(model_name="vgg16", num_classes=2, init_weights=True).to(device)
    #model = shufflenet_v2_x1_0(num_classes=2).to(device)
    
    #model_weight_path = "./shufflenet_v2_weights/best_model.pth"
    model_weight_path = "./efficentb0_weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    model.eval()
    
    for i in range(5):
        s = time.time()
        pred = model(img.to(device))
        pred = torch.nn.functional.softmax(pred)
        pred = pred.to("cpu")
        cls = torch.max(pred, dim=1)[1]
        print(time.time()-s)

    print("Prediction : {}".format(class_names[int(cls)]))
    print("Score : {}".format(float(pred[0][int(cls)])))

if __name__ == "__main__":
    main()
