import os
import argparse
import torch
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import MyDataSet, read_split_data, train_one_epoch, evaluate
import torch.nn as nn
from model import convnext_xlarge as create_model


def main(args):
    # to ensure that random results can be repeated
    random.seed(args.randomseed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tb_writer = SummaryWriter()
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    if os.path.exists(args.weights_savepath) is False: os.makedirs(args.weights_savepath)
    train_images_path, train_images_label, val_images_path, val_images_label=read_split_data(args.root, args.train_rate, args.randomseed)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop((args.img_size, args.img_size)),
                                    transforms.RandomHorizontalFlip(0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((int(args.img_size), int(args.img_size))),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)
                                               
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=train_dataset.collate_fn)
    if args.pretra:
        model = create_model(pretrained=args.online_weight, in_22k=True, num_classes=21841).to(device) 
        if not args.online_weight:
            model.load_state_dict(torch.load("weights/convnext_xlarge_22k_224.pth", map_location=device)["model"], strict=True)
        new_output_layer = nn.Linear(in_features=model.head.in_features, out_features=args.num_classes).to(device)
        model.head = new_output_layer    
    else:
        model = create_model(num_classes=args.num_classes).to(device)   

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1, verbose=False)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        model_weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        model.load_state_dict(model_weights_dict, strict=True)
        if "optimizer" in weights_dict:
            optimizer.load_state_dict(weights_dict['optimizer'])
        if "scheduler" in weights_dict:
            scheduler.load_state_dict(weights_dict['scheduler'])
        if "epoch" in weights_dict:
            args.pre_epochs = weights_dict['epoch'] 
            print('Load epoch {} successfully, continue training!'.format(args.pre_epochs))
    
    best_acc = 0.
    for epoch in range(args.pre_epochs+1, args.pre_epochs + args.epochs+1):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)      
        scheduler.step()
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        weights_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'scheduler': scheduler.state_dict()}
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = args.weights_savepath + './' + 'best_model.pth'
            torch.save(weights_dict, best_path)
        latest_path = args.weights_savepath + './' + 'latest_model.pth'
        torch.save(weights_dict, latest_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--pre_epochs', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--train_rate', type=float, default=0.9)
    parser.add_argument('--randomseed', type=int, default=42)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. cuda:0,1 or cpu)')
    parser.add_argument('--root', type=str, default="train_label.txt")
    parser.add_argument('--pretra', default=True, action='store_false')
    parser.add_argument('--online_weight', default=False, action='store_true')
    parser.add_argument('--weights_savepath', type=str, default="./weights")
    # Pre-train weight path. If there is no pre-training model, set to null character.
    parser.add_argument('--weights', type=str, default="weights/best_model.pth")
    opt = parser.parse_args()
    main(opt)
