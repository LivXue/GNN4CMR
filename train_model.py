from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import time
import copy
from evaluate import fx_calc_map_label
import numpy as np

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
criterion = nn.MultiLabelSoftMarginLoss()
criterion_md = nn.CrossEntropyLoss()


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim


def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha):
    term1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean() + ((view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()
    # term1 = criterion(view1_predict, labels_1) + criterion(view2_predict, labels_2)

    cos = lambda x, y: x.mm(y.t()) / (
        (x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    Sim22 = calc_label_sim(labels_2, labels_2).float()
    term21 = ((1+torch.exp(theta11)).log() - Sim11 * theta11).mean()
    term22 = ((1+torch.exp(theta12)).log() - Sim12 * theta12).mean()
    term23 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    term2 = term21 + term22 + term23

    im_loss = term1 + alpha * term2
    return im_loss


def calc_gan_loss(view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2, bs):
    img_md = torch.ones(bs, dtype=torch.long).cuda()
    txt_md = torch.zeros(bs, dtype=torch.long).cuda()
    return criterion_md(view1_modal_view1, img_md) + criterion_md(view2_modal_view1, txt_md) + \
           criterion_md(view1_modal_view2, img_md) + criterion_md(view2_modal_view2, txt_md)


def train_model(model, data_loaders, optimizer, alpha, beta, num_epochs=500):
    since = time.time()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for imgs, txts, labels in data_loaders[phase]:
                if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.float().cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    view1_feature, view2_feature, view1_predict, view2_predict, _, \
                        view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2 = model(imgs, txts)

                    bs = labels.shape[0]
                    gan_loss = calc_gan_loss(view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2, bs)

                    loss = calc_loss(view1_feature, view2_feature, view1_predict, view2_predict,
                                     labels, labels, alpha) + beta * gan_loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            if phase == 'train':
                print('Train Loss: {:.7f}'.format(epoch_loss))
            if phase == 'test':
                t_imgs, t_txts, t_labels = [], [], []
                img_md_img, text_md_img, img_md_text, text_md_text = 0, 0, 0, 0
                with torch.no_grad():
                    for imgs, txts, labels in data_loaders['test']:
                        if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            txts = txts.cuda()
                            labels = labels.float().cuda()
                        t_view1_feature, t_view2_feature, _, _, _, \
                            view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2 = model(imgs, txts)
                        t_imgs.append(t_view1_feature.cpu().numpy())
                        t_txts.append(t_view2_feature.cpu().numpy())
                        t_labels.append(labels.cpu().numpy())
                        bs = labels.shape[0]
                        img_md = torch.ones(bs, dtype=torch.long).cuda()
                        txt_md = torch.zeros(bs, dtype=torch.long).cuda()
                        img_md_img += torch.sum(torch.argmax(view1_modal_view1, dim=1) == img_md).cpu()
                        text_md_img += torch.sum(torch.argmax(view2_modal_view1, dim=1) == txt_md).cpu()
                        img_md_text += torch.sum(torch.argmax(view1_modal_view2, dim=1) == img_md).cpu()
                        text_md_text += torch.sum(torch.argmax(view2_modal_view2, dim=1) == txt_md).cpu()


                t_imgs = np.concatenate(t_imgs)
                t_txts = np.concatenate(t_txts)
                t_labels = np.concatenate(t_labels)

                img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
                txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)

                ds_len = float(len(data_loaders[phase].dataset))

                print('{} Loss: {:.7f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))
                print('Modal precision: Img2Img: {:.4f}, Txt2Img: {:.4f}, Img2Txt: {:.4f}, Txt2Txt: {:.4f}'.format(
                    img_md_img / ds_len, text_md_img / ds_len, img_md_text / ds_len, text_md_text / ds_len
                ))

            # deep copy the model
            if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
                best_acc = (img2text + txt2img) / 2.
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                test_img_acc_history.append(img2text)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, test_img_acc_history, test_txt_acc_history, epoch_loss_history


def train_model_incomplete(model, data_loaders, optimizer, alpha, beta, num_epochs=500):
    since = time.time()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0

            # zero the parameter gradients
            optimizer.zero_grad()

            # Iterate over data.
            if phase == 'train':
                for mul_data, img_data, txt_data in zip(data_loaders['train_complete'], data_loaders['train_img'], data_loaders['train_text']):

                    mul_imgs, mul_txts, mul_labels = mul_data
                    single_imgs, single_img_labels = img_data
                    single_txts, single_txt_labels = txt_data
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if torch.cuda.is_available():
                            mul_imgs = mul_imgs.cuda()
                            mul_txts = mul_txts.cuda()
                            mul_labels = mul_labels.float().cuda()
                            single_imgs = single_imgs.cuda()
                            single_img_labels = single_img_labels.float().cuda()
                            single_txts = single_txts.cuda()
                            single_txt_labels = single_txt_labels.float().cuda()

                        # Reconstruct modals
                        rescon_txts = model.img2text_net(model.img_net(single_imgs)).detach()
                        rescon_imgs = model.text2img_net(model.text_net(single_txts)).detach()
                        imgs = torch.cat([mul_imgs, single_imgs, rescon_imgs])
                        txts = torch.cat([mul_txts, rescon_txts, single_txts])
                        labels = torch.cat([mul_labels, single_img_labels, single_txt_labels])

                        if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                            print("Data contains Nan.")

                        # Forward
                        view1_feature, view2_feature, view1_predict, view2_predict, _, \
                            view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2 = model(imgs, txts)

                        bs = labels.shape[0]
                        gan_loss = calc_gan_loss(view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2, bs)

                        loss = calc_loss(view1_feature, view2_feature, view1_predict, view2_predict,
                                        labels, labels, alpha) + beta * gan_loss

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item()

                epoch_loss = running_loss / len(data_loaders['train_complete'].dataset)
                print('Train Loss: {:.7f}'.format(epoch_loss))
            if phase == 'test':
                t_imgs, t_txts, t_labels = [], [], []
                img_md_img, text_md_img, img_md_text, text_md_text = 0, 0, 0, 0
                with torch.no_grad():
                    for imgs, txts, labels in data_loaders['test']:
                        if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            txts = txts.cuda()
                            labels = labels.float().cuda()
                        view1_feature, view2_feature, view1_predict, view2_predict, _, \
                            view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2 = model(imgs, txts)
                        bs = labels.shape[0]
                        gan_loss = calc_gan_loss(view1_modal_view1, view2_modal_view1, view1_modal_view2,
                                                 view2_modal_view2, bs)
                        loss = calc_loss(view1_feature, view2_feature, view1_predict, view2_predict,
                                         labels, labels, alpha) + beta * gan_loss
                        running_loss += loss.item()
                        t_imgs.append(view1_feature.cpu().numpy())
                        t_txts.append(view2_feature.cpu().numpy())
                        t_labels.append(labels.cpu().numpy())
                        bs = labels.shape[0]
                        img_md = torch.ones(bs, dtype=torch.long).cuda()
                        txt_md = torch.zeros(bs, dtype=torch.long).cuda()
                        img_md_img += torch.sum(torch.argmax(view1_modal_view1, dim=1) == img_md).cpu()
                        text_md_img += torch.sum(torch.argmax(view2_modal_view1, dim=1) == txt_md).cpu()
                        img_md_text += torch.sum(torch.argmax(view1_modal_view2, dim=1) == img_md).cpu()
                        text_md_text += torch.sum(torch.argmax(view2_modal_view2, dim=1) == txt_md).cpu()


                t_imgs = np.concatenate(t_imgs)
                t_txts = np.concatenate(t_txts)
                t_labels = np.concatenate(t_labels)

                img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
                txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)

                ds_len = float(len(data_loaders[phase].dataset))
                epoch_loss = running_loss / len(data_loaders[phase].dataset)

                print('{} Loss: {:.7f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))
                print('Modal precision: Img2Img: {:.4f}, Txt2Img: {:.4f}, Img2Txt: {:.4f}, Txt2Txt: {:.4f}'.format(
                    img_md_img / ds_len, text_md_img / ds_len, img_md_text / ds_len, text_md_text / ds_len
                ))

            # deep copy the model
            if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
                best_acc = (img2text + txt2img) / 2.
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                test_img_acc_history.append(img2text)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, test_img_acc_history, test_txt_acc_history, epoch_loss_history