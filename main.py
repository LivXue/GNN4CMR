import os

import torch
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
from model import P_GNN, I_GNN
from train_model import train_model, train_model_incomplete
from load_data import get_loader
from evaluate import fx_calc_map_label

######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    dataset = 'NUS-WIDE-TC21'
    embedding = 'glove'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters
    DATA_DIR = 'data/' + dataset + '/'
    EVAL = False
    INCOMPLETE = True

    if dataset == 'mirflickr':
        alpha = 0.2
        beta = 0.2
        max_epoch = 40
        batch_size = 100
        lr = 5e-5
        lr2 = 4e-6
        betas = (0.5, 0.999)
        t = 0.4
        gnn = 'GCN'
        n_layers = 5
        k = 5
    elif dataset == 'NUS-WIDE-TC21':
        alpha = 0.2
        beta = 0.2
        max_epoch = 40
        batch_size = 1024
        lr = 5e-5
        lr2 = 4e-6
        betas = (0.5, 0.999)
        t = 0.4
        gnn = 'GCN'
        n_layers = 4
        k = 5
    elif dataset == 'MS-COCO':
        alpha = 0.2
        beta = 0.2
        max_epoch = 40
        batch_size = 1024
        lr = 5e-5
        lr2 = 1e-6
        betas = (0.5, 0.999)
        t = 0.4
        gnn = 'GCN'
        n_layers = 4
        k = 5
    else:
        raise NameError("Invalid dataset name!")

    seed = 103
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if embedding == 'glove':
        inp = loadmat('embedding/' + dataset + '-inp-glove6B.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'googlenews':
        inp = loadmat('embedding/' + dataset + '-inp-googlenews.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif embedding == 'fasttext':
        inp = loadmat('embedding/' + dataset + '-inp-fasttext.mat')['inp']
        inp = torch.FloatTensor(inp)
    else:
        inp = None

    print('...Data loading is beginning...')

    data_loader, input_data_par = get_loader(DATA_DIR, batch_size, False)

    print('...Data loading is completed...')

    model_ft = I_GNN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],
                     num_classes=input_data_par['num_class'], t=t, k=k, inp=inp, GNN=gnn, n_layers=n_layers).cuda()
    #model_ft = P_GNN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],
    #                 num_classes=input_data_par['num_class'], t=t, adj_file='data/' + dataset + '/adj.mat', inp=inp,
    #                 GNN=gnn, n_layers=n_layers).cuda()
    params_to_update = list(model_ft.parameters())

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)
    if EVAL:
        model_ft.load_state_dict(torch.load('model/DALGCN_' + dataset + '.pth'))
    else:
        print('...Training is beginning...')
        # Train and evaluate
        if INCOMPLETE:
            model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model_ft, data_loader, optimizer, alpha, beta, max_epoch)
            data_loader, input_data_par = get_loader(DATA_DIR, batch_size, True)
            optimizer = optim.Adam(params_to_update, lr=lr2, betas=betas)
            model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model_incomplete(model_ft, data_loader, optimizer, alpha, beta, max_epoch)
        else:
            model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model_ft, data_loader, optimizer, alpha, beta, max_epoch)
        print('...Training is completed...')

        torch.save(model_ft.state_dict(), 'model/DALGNN_' + dataset + '.pth')

    print('...Evaluation on testing data...')
    view1_feature, view2_feature, view1_predict, view2_predict, classifiers, _, _, _, _ = model_ft(
        torch.tensor(input_data_par['img_test']).cuda(), torch.tensor(input_data_par['text_test']).cuda())
    label = input_data_par['label_test']
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()
    img_to_txt = fx_calc_map_label(view1_feature, view2_feature, label)
    print('...Image to Text MAP = {}'.format(img_to_txt))

    txt_to_img = fx_calc_map_label(view2_feature, view1_feature, label)
    print('...Text to Image MAP = {}'.format(txt_to_img))

    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))
