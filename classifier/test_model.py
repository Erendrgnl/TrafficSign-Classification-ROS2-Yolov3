from dataset_loader import CustomImageDataset
from torch.utils.data import DataLoader
from models.efficentNet import efficientnet_b0
from models.shuffleNetv2 import shufflenet_v2_x1_0
from models.vgg import vgg
from torchvision import transforms
import torch

"""
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
"""
def main():
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    device = "cuda"

    data_transform = transforms.Compose([transforms.RandomResizedCrop(224)])
    test_data = CustomImageDataset("dataset/test",transform=data_transform)
    test_loader = DataLoader(dataset = test_data, batch_size = 1)

    #model = efficientnet_b0(num_classes=test_data.num_class).to(device)
    #model = vgg(model_name="vgg16", num_classes=test_data.num_class, init_weights=True).to(device)
    model = shufflenet_v2_x1_0(num_classes=test_data.num_class).to(device)
    model_weight_path = "./shufflenet_v2_weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    model.eval()
    
    total_num = len(test_loader.dataset)
    sum_num = torch.zeros(1).to(device)

    for images, labels in test_loader:
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        pred = pred.to("cpu")
        if(pred==0 and labels ==0):
            tp += 1
        elif(pred==1 and labels == 1):
            tn += 1
        elif(pred==0 and labels == 1):
            fp += 1
        elif(pred==1 and labels == 0):
            fn += 1   
       
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1_score = 2* ((precision * recall) / (precision+recall))
    test_acc = (tp+tn) / total_num
    print("Precision {}".format(precision))
    print("Recall {}".format(recall))
    print("f1_score {}".format(f1_score))
    print("test_acc {}".format(test_acc))


if __name__ == "__main__":
    main()
