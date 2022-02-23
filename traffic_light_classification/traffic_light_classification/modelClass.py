from torchvision import transforms
import torch
from PIL import Image
from traffic_light_classification.efficentNet import efficientnet_b0

class TrafficLightClassifier(object):
    def __init__(self,device):
        self.device = device
        self.class_names = ["go","stop"]

        self.transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.ToTensor()])

        model = efficientnet_b0(num_classes=2).to(device)
        model_weight_path = "/home/eren/best_model.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()

        self.model = model

    def prediction(self,img):
        img = Image.fromarray(img)
        img = self.transform(img)
        img = img.view((1,3,224,224))
        pred = self.model(img.to(self.device))
        pred = torch.nn.functional.softmax(pred)
        pred = pred.to("cpu")
        cls_id = torch.max(pred, dim=1)[1]
        cls_name = self.class_names[int(cls_id)]
        score = pred[0][int(cls_id)]
        return cls_name,float(score)