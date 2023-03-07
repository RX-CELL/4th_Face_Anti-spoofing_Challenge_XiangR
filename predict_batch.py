import argparse
import os
import torch
import pandas as pd
from torchvision import transforms
from utils import MyDataSet
from model import convnext_xlarge as create_model
from tqdm import tqdm
import torch.nn.functional as F 
def predict_batch(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop((args.img_size, args.img_size)),
                                    transforms.RandomHorizontalFlip(0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((int(args.img_size), int(args.img_size))),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    path = pd.read_csv(args.test_txt, sep=' ', header=None, names=['img_path'])
    dev_images_path = list(path.iloc[:,0])
    dev_images_label = [9]*len(dev_images_path)
    dev_dataset = MyDataSet(images_path=dev_images_path,
                            images_class=dev_images_label,
                            transform=data_transform["val"])
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dev_loader = torch.utils.data.DataLoader(dev_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=dev_dataset.collate_fn)
    print('Using {} dataloader workers every process'.format(nw))
    model = create_model(num_classes=args.num_classes).to(device) 
    weights_dict = torch.load(args.weights, map_location=device)
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict, strict=True)
    
    model.eval()
    lab_pre = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in tqdm(dev_loader):
            X= X.to(device)
            pred = model(X)
            pre_soft = F.softmax(pred, dim=1).detach().cpu()
            lab_pre_temp = torch.cat((pred.argmax(1).detach().cpu().reshape(-1, 1), pre_soft[:, 1].reshape(-1, 1)), dim=1)
            lab_pre = torch.cat((lab_pre, lab_pre_temp))
    lab_pre = lab_pre.numpy()
    lab_pre = pd.DataFrame(data=lab_pre, columns=['pred', 'pre_class1'])
    result = pd.concat([path, lab_pre.iloc[:,1]],axis=1)
    result.to_csv(args.pre_txt, sep=' ', index=None, header=None)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--weights', type=str, default="weights/best_model.pth")
    parser.add_argument('--test_txt', type=str, default="dev.txt")
    parser.add_argument('--pre_txt', type=str, default="phase1.txt")
    opt = parser.parse_args()
    for orgtxt, pretxt in zip(["dev.txt", "test.txt"], ["phase1.txt", "pre_test.txt"]):
        opt.test_txt, opt.pre_txt = orgtxt, pretxt
        predict_batch(opt)
    # Concatenate phase1.txt and pre_test.txt to get phase2.txt
    phase1=pd.read_csv("phase1.txt", header=None, sep=' ', index_col=False)
    pre_test=pd.read_csv("pre_test.txt", header=None, sep=' ', index_col=False)
    phase2=pd.concat([phase1, pre_test])
    phase2.to_csv("phase2.txt", header=None, index=False, sep=' ')
    print(phase1.shape, pre_test.shape, phase2.shape)
    

