import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
import cv2
import numpy as np
import time
from Edgeloss import ECELoss
from MSDS_ResNet import Res_UNet

class ImgDataset_UVs(Dataset):
    def __init__(self, imgs_root_dir, imgs_4X_root_dir, imgs_8X_root_dir, labels_root_dir, transform=None):
        self.imgs_root_dir = imgs_root_dir
        self.imgs_4X_root_dir = imgs_4X_root_dir
        self.imgs_8X_root_dir = imgs_8X_root_dir
        self.labels_root_dir = labels_root_dir
        self.transform = transform
        self.imgs = os.listdir(self.imgs_root_dir)
        self.imgs_4X = os.listdir(self.imgs_4X_root_dir)
        self.imgs_8X = os.listdir(self.imgs_8X_root_dir)
        self.labels = os.listdir(self.labels_root_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        img_idx = self.imgs[idx]
        img_4X_idx = self.imgs_4X[idx]
        img_8X_idx = self.imgs_8X[idx]
        label_idx = self.labels[idx]

        img_path = os.path.join(self.imgs_root_dir, img_idx)
        img_4X_path = os.path.join(self.imgs_4X_root_dir, img_4X_idx)
        img_8X_path = os.path.join(self.imgs_8X_root_dir, img_8X_idx)
        label_path = os.path.join(self.labels_root_dir, label_idx)

        img_rgb = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        img_4X_rgb = cv2.imread(img_4X_path, cv2.COLOR_BGR2RGB)
        img_8X_rgb = cv2.imread(img_8X_path, cv2.COLOR_BGR2RGB)
        label_gray = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        label_gray[label_gray == 255] = 0

        if self.transform:
            img_rgb = self.transform(img_rgb)
            img_4X_rgb = self.transform(img_4X_rgb)
            img_8X_rgb = self.transform(img_8X_rgb)

        return img_rgb, img_4X_rgb, img_8X_rgb, label_gray

class ImgDataset_UVs_files(Dataset):
    def __init__(self, imgs_urls, imgs_root_dir, imgs_4X_root_dir, imgs_8X_root_dir, labels_root_dir, transform=None):
        self.imgs_root_dir = imgs_root_dir
        self.imgs_4X_root_dir = imgs_4X_root_dir
        self.imgs_8X_root_dir = imgs_8X_root_dir
        self.labels_root_dir = labels_root_dir
        self.transform = transform
        self.imgs = os.listdir(self.imgs_root_dir)
        self.imgs_4X = os.listdir(self.imgs_4X_root_dir)
        self.imgs_8X = os.listdir(self.imgs_8X_root_dir)
        self.labels = os.listdir(self.labels_root_dir)

        self.fnames = [line.strip() for line in open(imgs_urls, "r", encoding="utf-8")]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):

        img_name = self.fnames[idx]

        img_idx = img_name + ".tif"
        img_4X_idx = img_name + ".tif"
        img_8X_idx = img_name + ".tif"
        label_idx = img_name + ".tif"

        img_path = os.path.join(self.imgs_root_dir, img_idx)
        img_4X_path = os.path.join(self.imgs_4X_root_dir, img_4X_idx)
        img_8X_path = os.path.join(self.imgs_8X_root_dir, img_8X_idx)
        label_path = os.path.join(self.labels_root_dir, label_idx)

        img_rgb = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        img_4X_rgb = cv2.imread(img_4X_path, cv2.COLOR_BGR2RGB)
        img_8X_rgb = cv2.imread(img_8X_path, cv2.COLOR_BGR2RGB)
        label_gray = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        label_gray[label_gray == 255] = 0

        if self.transform:
            img_rgb = self.transform(img_rgb)
            img_4X_rgb = self.transform(img_4X_rgb)
            img_8X_rgb = self.transform(img_8X_rgb)

        return img_rgb, img_4X_rgb, img_8X_rgb, label_gray

def adjust_lr(optimizer, step, initial_lr):
    lr = initial_lr * ((1 - float(step) / 60) ** 0.9)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def iou_mean(pred, target, n_classes = 1):
    ious = []
    iousSum = 0
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = 0
    union = 0

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return intersection, union

if __name__ == "__main__":

    train_imgs_Path = "Train_image"
    train_imgs_4X_Path = "Train_image_4X"
    train_imgs_8X_Path = "Train_image_8X"
    train_labels_Path = "Train_image_label"

    # train_fname = "train.csv"
    # test_fname = "validation.csv"
    # train_dataset = ImgDataset_UVs_files(imgs_urls=train_fname, imgs_root_dir=train_imgs_Path,
    #                                      imgs_4X_root_dir=train_imgs_4X_Path, imgs_8X_root_dir=train_imgs_8X_Path,
    #                                      labels_root_dir=train_labels_Path, transform=transforms.ToTensor())
    # test_dataset = ImgDataset_UVs_files(imgs_urls=test_fname, imgs_root_dir=train_imgs_Path,
    #                                     imgs_4X_root_dir=train_imgs_4X_Path, imgs_8X_root_dir=train_imgs_8X_Path,
    #                                     labels_root_dir=train_labels_Path, transform=transforms.ToTensor())

    All_dataset = ImgDataset_UVs(imgs_root_dir=train_imgs_Path, imgs_4X_root_dir=train_imgs_4X_Path,
                                 imgs_8X_root_dir=train_imgs_8X_Path, labels_root_dir=train_labels_Path,
                                 transform=transforms.ToTensor())
    train_size = int(len(All_dataset) * 0.7)
    test_size = int(len(All_dataset) * 0.3)
    train_dataset, test_dataset = random_split(All_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_idx in range(1, 11):
        loss_history_train = []
        loss_history_test = []
        iou_history_test = []
        print("Model %d Training..." % model_idx)
        model_path = "models/MSDS_" + str(model_idx) + ".pt"
        loss_iou_path = "loss/loss_iou_MSDS_" + str(model_idx) + ".csv"
        uv_model = Res_UNet(n_channels=3, pretrained=False, n_classes=2).to(device)

        criterion = ECELoss(n_classes=2)
        optimizer = optim.Adam(uv_model.parameters(), lr=0.0001)
        min_loss = 10.0
        min_IoU = 0.0

        for epoch in range(20):
            print('\nEpoch: %d' % (epoch + 1))
            uv_model.train()
            sum_loss = 0.0
            inter = 0
            unin = 0
            sum_mIoU = 0.0
            start_time = time.time()
            for i, data in enumerate(train_loader, 0):
                # prepare dataset
                images, images_4X, images_8X, labels = data
                images, images_4X, images_8X, labels = images.to(device), images_4X.to(device), images_8X.to(
                    device), labels.to(device)
                optimizer.zero_grad()

                # forward & backward
                outputs = uv_model(images, images_4X, images_8X)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                # print ac & loss in each batch
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                it_batch, un_batch = iou_mean(predicted, labels, n_classes=1)
                inter += it_batch
                unin += un_batch
            sum_mIoU = inter * 1.0 / unin
            print('[epoch:%d] Loss: %.03f | mIoU: %.03f'% (epoch + 1, sum_loss / len(train_loader), sum_mIoU))
            loss_history_train.append((sum_loss / len(train_loader)))

            # get the ac with testdataset in each epoch
            print('Waiting Test...')
            uv_model.eval()
            with torch.no_grad():
                cal_mIoU = 0.0
                inter = 0
                unin = 0
                Aver_loss = 0.0
                sum_loss = 0.0
                for data in test_loader:
                    images, images_4X, images_8X, labels = data
                    images, images_4X, images_8X, labels = images.to(device), images_4X.to(device), images_8X.to(
                        device), labels.to(device)
                    outputs = uv_model(images, images_4X, images_8X)
                    loss = criterion(outputs, labels.long())
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    it_batch, un_batch = iou_mean(predicted, labels, n_classes=1)
                    inter += it_batch
                    unin += un_batch
                Aver_loss = sum_loss / len(test_loader)
                cal_mIoU = inter * 1.0 / unin
                print('Test\'s mIoU is: %.03f' % (cal_mIoU))
                print('Test\'s loss is: %.03f' % (Aver_loss))
                loss_history_test.append(Aver_loss)
                iou_history_test.append(cal_mIoU)
            if cal_mIoU > min_IoU:
                min_IoU = cal_mIoU
                print("Model Saving...")
                torch.save(uv_model.state_dict(), model_path)
            else:
                print("IoU did not improve...")
                print("the best mIoU is", min_IoU)
            end_time = time.time()
            adjust_lr(optimizer, (epoch + 1), 0.0001)
            print("Spending " + str(end_time - start_time) + "s")
        loss_history_train, loss_history_test, iou_history_test = np.array(loss_history_train), np.array(
            loss_history_test), np.array(iou_history_test)
        writeinfile = np.vstack((loss_history_train, loss_history_test, iou_history_test))
        np.savetxt(loss_iou_path, writeinfile, delimiter=",")