import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import queue
import time

import numpy as np
import torch
import torch.fft as fft
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from ts_data.preprocessing import fill_nan_value
from ts_data.dataloader import UCRDataset, UCRTFAugDataset
from ts_model.loss import sup_contrastive_loss, reg_co_training_loss, SimCLRContrastiveLoss
from ts_model.model import ProjectionHead, base_Model, ClassifierLogit
from utils.ts_utils import set_seed, build_dataset, get_all_datasets, \
    construct_graph_via_knn_cpl_nearind_gpu, \
    build_loss, shuffler, evaluate, convert_coeff
from utils.aug_utils import get_freq_augmentation, get_time_augmentation


def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels


def build_dataset_pt(args):
    data_path = f"./{args.dataset}"
    train_dataset_dict = torch.load(os.path.join(data_path, "train.pt"))
    train_dataset = train_dataset_dict["samples"].numpy()  # (num_size, num_dimensions, series_length)
    train_target = train_dataset_dict["labels"].numpy()
    num_classes = len(np.unique(train_dataset_dict["labels"].numpy(), return_counts=True)[0])
    train_target = transfer_labels(train_target)

    val_dataset_dict = torch.load(os.path.join(data_path, "val.pt"))
    val_dataset = val_dataset_dict["samples"].numpy() 
    val_target = val_dataset_dict["labels"].numpy()
    val_target = transfer_labels(val_target)

    test_dataset_dict = torch.load(os.path.join(data_path, "test.pt"))
    test_dataset = test_dataset_dict["samples"].numpy()
    test_target = test_dataset_dict["labels"].numpy()
    test_target = transfer_labels(test_target)

    return train_dataset, train_target, val_dataset, val_target, test_dataset, test_target, num_classes


dataset_para = {
    'HAR': {'features_len': 18, 'features_len_freq': 10},
    'Epilepsy': {'features_len': 24, 'features_len_freq': 13},
    'SleepEDF': {'features_len': 377, 'features_len_freq': 190},
    'Wafer': {'features_len': 21, 'features_len_freq': 12},
    'FordA': {'features_len': 65, 'features_len_freq': 34},
    'FordB': {'features_len': 65, 'features_len_freq': 34},
    'PhalangesOutlinesCorrect': {'features_len': 12, 'features_len_freq': 7},
    'ProximalPhalanxOutlineCorrect': {'features_len': 12, 'features_len_freq': 7},
    'StarLightCurves': {'features_len': 130, 'features_len_freq': 66},
    'ElectricDevices': {'features_len': 14, 'features_len_freq': 8},
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Base setup
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')
    parser.add_argument('--features_len', type=int, default=25, help='')
    parser.add_argument('--kernel_size', type=int, default=8, help='')
    parser.add_argument('--final_out_channels', type=int, default=128, help='')
    parser.add_argument('--stride', type=int, default=1, help='')
    parser.add_argument('--dropout', type=int, default=0.35, help='')
    parser.add_argument('--input_channels', type=int, default=1, help='')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='HAR', help='dataset') 
    parser.add_argument('--dataroot', type=str, default='./UCRArchive_2018', help='path of UCR folder')
    parser.add_argument('--num_classes', type=int, default=0, help='number of class')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')

    # Semi training
    parser.add_argument('--labeled_ratio', type=float, default=0.05, help='0.01, 0.05')
    parser.add_argument('--warmup_epochs', type=int, default=40, help='warmup epochs using only labeled data for ssc')
    parser.add_argument('--queue_maxsize', type=int, default=3, help='2 or 3')
    parser.add_argument('--knn_num_tem', type=int, default=40, help='10, 20, 50')
    parser.add_argument('--knn_num_feq', type=int, default=30, help='10, 20, 50')

    ## Augmentation
    parser.add_argument('--aug_time_method_index', type=int, default=1, help='1,2,3,4,5,6')
    parser.add_argument('--aug2_time_method_index', type=int, default=3, help='1,2,3,4,5,6')
    parser.add_argument('--aug_freq_method_index', type=int, default=1, help='1,2,3,4')
    parser.add_argument('--aug2_freq_method_index', type=int, default=2, help='1,2,3,4')
    parser.add_argument('--self_contra_tau', type=float, default=10, help='0.1') 

    # Contrastive loss
    parser.add_argument('--sup_con_mu', type=float, default=0.05, help='0.05 or 0.005')
    parser.add_argument('--sup_con_lamda', type=float, default=0.05, help='0.05 or 0.005')
    parser.add_argument('--mlp_head', type=bool, default=True, help='head project')
    parser.add_argument('--temperature', type=float, default=50, help='20, 50')

    # training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
    parser.add_argument('--epoch', type=int, default=80, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    # classifier setup
    parser.add_argument('--classifier', type=str, default='linear', help='')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)
    
    if args.dataset == 'HAR' or args.dataset == 'Epilepsy' or args.dataset == 'SleepEDF':
        train_dataset, train_target, val_dataset, val_target, test_dataset, test_target, num_classes = build_dataset_pt(args)
    else:
        sum_dataset, sum_target, num_classes = build_dataset(args)
        train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(sum_dataset, sum_target)
        train_dataset = train_datasets[0]
        train_target = train_targets[0]
        val_dataset = val_datasets[0]
        val_target = val_targets[0]
        test_dataset = test_datasets[0]
        test_target = test_targets[0]
        train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)
        train_dataset = np.expand_dims(train_dataset, axis=1)
        val_dataset   = np.expand_dims(val_dataset, axis=1)
        test_dataset  = np.expand_dims(test_dataset, axis=1)
        
    args.num_classes = num_classes
    args.seq_len = train_dataset.shape[2]
    args.input_size = train_dataset.shape[1]
    while train_dataset.shape[0] < args.batch_size:
        args.batch_size = args.batch_size // 2

    if args.batch_size * 2 > train_dataset.shape[0]:
        args.queue_maxsize = 2
        
    
    ## Time Domain Settings
    args.features_len = dataset_para[args.dataset]['features_len']
    args.input_channels = args.input_size
    model = base_Model(args)
    classifier = ClassifierLogit(input_dims=args.features_len * args.final_out_channels, output_dims=args.num_classes)
    projection_head = ProjectionHead(input_dim=args.features_len * args.final_out_channels)
    projection_head_aug1 = ProjectionHead(input_dim=args.features_len * args.final_out_channels)
    projection_head_aug2 = ProjectionHead(input_dim=args.features_len * args.final_out_channels)

    model, classifier = model.to(device), classifier.to(device)
    projection_head = projection_head.to(device)
    projection_head_aug1 = projection_head_aug1.to(device)
    projection_head_aug2 = projection_head_aug2.to(device)

    loss = build_loss(args).to(device)
    smclr_criterion = SimCLRContrastiveLoss(temperature=args.self_contra_tau)

    model_init_state = model.state_dict()
    classifier_init_state = classifier.state_dict()
    projection_head_init_state = projection_head.state_dict()
    projection_head_aug1_init_state = projection_head_aug1.state_dict()
    projection_head_aug2_init_state = projection_head_aug2.state_dict()
    
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': classifier.parameters()},
                                  {'params': projection_head.parameters()},
                                  {'params': projection_head_aug1.parameters()},
                                  {'params': projection_head_aug2.parameters()}],
                                 lr=args.lr)
    
    ## Frequency Domain Settings
    args.input_size = 2 * train_dataset.shape[1]
    args.input_channels = args.input_size
    args.features_len = dataset_para[args.dataset]['features_len_freq']
    model_feq = base_Model(args)
    classifier_feq = ClassifierLogit(input_dims=args.features_len * args.final_out_channels, output_dims=args.num_classes)
    projection_head_feq = ProjectionHead(input_dim=args.features_len * args.final_out_channels)
    projection_head_feq_aug1 = ProjectionHead(input_dim=args.features_len * args.final_out_channels)
    projection_head_feq_aug2 = ProjectionHead(input_dim=args.features_len * args.final_out_channels)

    model_feq, classifier_feq = model_feq.to(device), classifier_feq.to(device)
    projection_head_feq = projection_head_feq.to(device)
    projection_head_feq_aug1 = projection_head_feq_aug1.to(device)
    projection_head_feq_aug2 = projection_head_feq_aug2.to(device)

    loss_feq = build_loss(args).to(device)
    smclr_criterion_freq = SimCLRContrastiveLoss(temperature=args.self_contra_tau)
    
    model_feq_init_state = model_feq.state_dict()
    classifier_feq_init_state = classifier_feq.state_dict()
    projection_head_feq_init_state = projection_head_feq.state_dict()
    projection_head_feq_aug1_init_state = projection_head_feq_aug1.state_dict()
    projection_head_feq_aug2_init_state = projection_head_feq_aug2.state_dict()

    is_projection_head = args.mlp_head

    optimizer_feq = torch.optim.Adam(
        [{'params': model_feq.parameters()}, {'params': classifier_feq.parameters()},
         {'params': projection_head_feq.parameters()},
         {'params': projection_head_feq_aug1.parameters()},
         {'params': projection_head_feq_aug2.parameters()}],
        lr=args.lr)

    print('Start SSC on the {}'.format(args.dataset))

    losses = []
    test_accuracies = []
    train_time = 0.0
    end_val_epochs = []

    test_accuracies_tem = []
    end_val_epochs_tem = []
    test_accuracies_feq = []
    end_val_epochs_feq = []

    for i in range(1):
        t = time.time()
        
        ## Time Domain
        model.load_state_dict(model_init_state)
        classifier.load_state_dict(classifier_init_state)
        projection_head.load_state_dict(projection_head_init_state)
        projection_head_aug1.load_state_dict(projection_head_aug1_init_state)
        projection_head_aug2.load_state_dict(projection_head_aug2_init_state)
        
        ## Frequency Domain
        model_feq.load_state_dict(model_feq_init_state)
        classifier_feq.load_state_dict(classifier_feq_init_state)
        projection_head_feq.load_state_dict(projection_head_feq_init_state)
        projection_head_feq_aug1.load_state_dict(projection_head_feq_aug1_init_state)
        projection_head_feq_aug2.load_state_dict(projection_head_feq_aug2_init_state)

        print("Start training and evaluate: ")

        train_labeled, train_unlabeled, y_labeled, y_unlabeled = train_test_split(train_dataset, train_target,
                                                                                  test_size=(1 - args.labeled_ratio),
                                                                                  random_state=args.random_seed)

        mask_labeled = np.zeros(len(y_labeled))
        mask_unlabeled = np.ones(len(y_unlabeled))
        mask_train = np.concatenate([mask_labeled, mask_unlabeled])
        train_all_split = np.concatenate([train_labeled, train_unlabeled])
        y_label_split = np.concatenate([y_labeled, y_unlabeled])

        x_train_all, y_train_all = shuffler(train_all_split, y_label_split)
        mask_train, _ = shuffler(mask_train, mask_train)
        y_train_all[mask_train == 1] = -1  ## Generate unlabeled data
        
        aug_x_train_all_list = []
        aug2_x_train_all_list = []

        num_channels = x_train_all.shape[1]

        for _i in range(num_channels):
            channel_data = x_train_all[:, _i, :]
            aug_channel = get_time_augmentation(
                train_dataset=channel_data,
                method_index=args.aug_time_method_index
            )
            
            aug2_channel = get_time_augmentation(
                train_dataset=channel_data,
                method_index=args.aug2_time_method_index
            )
            
            if len(aug_channel.shape) == 3 and aug_channel.shape[1] == 1:
                aug_channel = aug_channel.squeeze(1)
            if len(aug2_channel.shape) == 3 and aug2_channel.shape[1] == 1:
                aug2_channel = aug2_channel.squeeze(1)
            
            aug_x_train_all_list.append(aug_channel)
            aug2_x_train_all_list.append(aug2_channel)

        aug_x_train_all = np.stack(aug_x_train_all_list, axis=1)
        aug2_x_train_all = np.stack(aug2_x_train_all_list, axis=1)

        aug_x_train_all = torch.from_numpy(aug_x_train_all).type(torch.FloatTensor).to(device)
        aug2_x_train_all = torch.from_numpy(aug2_x_train_all).type(torch.FloatTensor).to(device)
        
        train_fft_list = []
        aug_x_freq_train_all_list = []
        aug2_x_freq_train_all_list = []

        num_channels = x_train_all.shape[1]
        
        for _j in range(num_channels):
            channel_data = x_train_all[:, _j, :]
            
            channel_tensor = torch.from_numpy(channel_data)
            
            fft_result = fft.rfft(channel_tensor, dim=-1)
            fft_converted, _ = convert_coeff(fft_result)
            train_fft_list.append(fft_converted)  # 移除通道维度
            
            aug_channel = get_freq_augmentation(
                torch_train_dataset=torch.from_numpy(channel_data),
                method_index=args.aug_freq_method_index
            )
            
            if args.aug_freq_method_index != 4:
                aug_channel = fft.rfft(aug_channel, dim=-1)
                aug_channel, _ = convert_coeff(aug_channel)
            
            aug_x_freq_train_all_list.append(aug_channel)
            aug2_channel = get_freq_augmentation(
                torch_train_dataset=torch.from_numpy(channel_data),
                method_index=args.aug2_freq_method_index
            )
            
            if args.aug2_freq_method_index != 4:
                aug2_channel = fft.rfft(aug2_channel, dim=-1)
                aug2_channel, _ = convert_coeff(aug2_channel)
            
            aug2_x_freq_train_all_list.append(aug2_channel)
        
        train_fft = torch.cat(train_fft_list, dim=1) 
        aug_x_freq_train_all = torch.cat(aug_x_freq_train_all_list, dim=1)  
        aug2_x_freq_train_all = torch.cat(aug2_x_freq_train_all_list, dim=1)  

        train_fft = train_fft.type(torch.FloatTensor).to(device)
        aug_x_freq_train_all = aug_x_freq_train_all.type(torch.FloatTensor).to(device)
        aug2_x_freq_train_all = aug2_x_freq_train_all.type(torch.FloatTensor).to(device)
        
        x_train_all = torch.from_numpy(x_train_all).type(torch.FloatTensor).to(device)
        y_train_all = torch.from_numpy(y_train_all).type(torch.FloatTensor).to(device).to(torch.int64)

        x_train_labeled_all = x_train_all[mask_train == 0]
        x_train_labeled_all_feq = train_fft[mask_train == 0]
        y_train_labeled_all = y_train_all[mask_train == 0]

        x_train_aug_labeled_all = aug_x_train_all[mask_train == 0]
        x_train_aug2_labeled_all = aug2_x_train_all[mask_train == 0]

        x_train_freq_aug_labeled_all = aug_x_freq_train_all[mask_train == 0]
        x_train_freq_aug2_labeled_all = aug2_x_freq_train_all[mask_train == 0]

        aug_train_set_labled = UCRTFAugDataset(dataset=x_train_labeled_all, freq_data=x_train_labeled_all_feq,
                                                  data_aug1=x_train_aug_labeled_all, data_aug2=x_train_aug2_labeled_all,
                                                  freq_aug1=x_train_freq_aug_labeled_all,
                                                  freq_aug2=x_train_freq_aug2_labeled_all,
                                                  target=y_train_labeled_all)
        aug_train_set = UCRTFAugDataset(dataset=x_train_all, freq_data=train_fft,
                                           data_aug1=aug_x_train_all, data_aug2=aug2_x_train_all,
                                           freq_aug1=aug_x_freq_train_all,
                                           freq_aug2=aug2_x_freq_train_all,
                                           target=y_train_all,
                                           mask_train=mask_train)

        val_set = UCRDataset(torch.from_numpy(val_dataset).type(torch.FloatTensor).to(device),
                             torch.from_numpy(val_target).type(torch.FloatTensor).to(device).to(torch.int64))
        test_set = UCRDataset(torch.from_numpy(test_dataset).type(torch.FloatTensor).to(device),
                              torch.from_numpy(test_target).type(torch.FloatTensor).to(device).to(torch.int64))

        batch_size_labeled = 128
        while x_train_labeled_all.shape[0] < batch_size_labeled:
            batch_size_labeled = batch_size_labeled // 2

        if x_train_labeled_all.shape[0] < 16:
            batch_size_labeled = 16

        train_labeled_loader = DataLoader(aug_train_set_labled, batch_size=batch_size_labeled, num_workers=0,
                                          drop_last=False)

        train_loader = DataLoader(aug_train_set, batch_size=args.batch_size, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)
        
        val_fft_list = []
        num_channels = val_dataset.shape[1]
        for _j in range(num_channels):
            channel_data = val_dataset[:, _j, :]
            channel_tensor = torch.from_numpy(channel_data)

            fft_result = fft.rfft(channel_tensor, dim=-1)
            fft_converted, _ = convert_coeff(fft_result)
            val_fft_list.append(fft_converted) 
            
        val_fft = torch.cat(val_fft_list, dim=1) 
        val_set_feq = UCRDataset(val_fft.type(torch.FloatTensor).to(device), torch.from_numpy(val_target).type(torch.FloatTensor).to(device).to(torch.int64))
        
        test_fft_list = []
        num_channels = test_dataset.shape[1]
        for _j in range(num_channels):
            channel_data = test_dataset[:, _j, :]
            channel_tensor = torch.from_numpy(channel_data)
            fft_result = fft.rfft(channel_tensor, dim=-1)
            fft_converted, _ = convert_coeff(fft_result)
            test_fft_list.append(fft_converted) 
            
        test_fft = torch.cat(test_fft_list, dim=1)  
        test_set_feq = UCRDataset(test_fft.type(torch.FloatTensor).to(device), torch.from_numpy(test_target).type(torch.FloatTensor).to(device).to(torch.int64))

        val_loader_feq = DataLoader(val_set_feq, batch_size=args.batch_size, num_workers=0)
        test_loader_feq = DataLoader(test_set_feq, batch_size=args.batch_size, num_workers=0)

        train_loss = []
        train_accuracy = []
        train_loss_feq = []
        train_accuracy_feq = []
        num_steps = args.epoch // args.batch_size

        last_loss = float('inf')
        stop_count = 0
        increase_count = 0

        num_steps = aug_train_set.__len__() // args.batch_size
        if num_steps == 0:
            num_steps = num_steps + 1

        min_val_loss = float('inf')
        test_accuracy = 0
        end_val_epoch = 0
        min_val_loss_feq = float('inf')
        test_accuracy_tem = 0
        end_val_epoch_tem = 0
        test_accuracy_feq = 0
        end_val_epoch_feq = 0

        queue_train_x = queue.Queue(args.queue_maxsize)
        queue_train_y = queue.Queue(args.queue_maxsize)
        queue_train_mask = queue.Queue(args.queue_maxsize)

        queue_train_x_feq = queue.Queue(args.queue_maxsize)
        queue_train_y_feq = queue.Queue(args.queue_maxsize)

        for epoch in range(args.epoch):

            epoch_train_loss = 0
            epoch_train_acc = 0
            num_iterations = 0

            model.train()
            classifier.train()
            projection_head.train()
            projection_head_aug1.train()
            projection_head_aug2.train()
            
            model_feq.train()
            classifier_feq.train()
            projection_head_feq.train()
            projection_head_feq_aug1.train()
            projection_head_feq_aug2.train()

            if epoch < args.warmup_epochs:
                for x, x_feq, x_aug1, x_aug2, freq_aug1, freq_aug2, y, _ in train_labeled_loader:
                    if x.shape[0] < 2:
                        continue

                    optimizer.zero_grad()
                    optimizer_feq.zero_grad()

                    pred_embed = model(x)
                    pred_embed_aug1 = model(x_aug1)
                    pred_embed_aug2 = model(x_aug2)

                    pred_embed_feq = model_feq(x_feq)
                    pred_embed_feq_aug1 = model_feq(freq_aug1)
                    pred_embed_feq_aug2 = model_feq(freq_aug2)

                    if is_projection_head:
                        preject_head_embed = projection_head(pred_embed)
                        preject_head_embed_feq = projection_head_feq(pred_embed_feq)

                    pred = classifier(pred_embed)
                    pred_aug1 = classifier(pred_embed_aug1)
                    pred_aug2 = classifier(pred_embed_aug2)

                    pred_feq = classifier_feq(pred_embed_feq)
                    pred_feq_aug1 = classifier_feq(pred_embed_feq_aug1)
                    pred_feq_aug2 = classifier_feq(pred_embed_feq_aug2)

                    hyp_epoch = (epoch * 1.0 / args.warmup_epochs)

                    reg_loss = reg_co_training_loss(time_pre=pred, freq_pre=pred_feq.detach()).to(device)
                    reg_loss_freq = reg_co_training_loss(time_pre=pred.detach(), freq_pre=pred_feq).to(device)
                    
                    step_loss = loss(pred, y) + hyp_epoch * (loss(pred_aug1, y) + loss(pred_aug2, y)) + reg_loss
                    step_loss_feq = loss_feq(pred_feq, y) + hyp_epoch * (
                            loss_feq(pred_feq_aug1, y) + loss_feq(pred_feq_aug2, y)) + reg_loss_freq 

                    if len(y) > 1:
                        batch_sup_contrastive_loss = sup_contrastive_loss(
                            embd_batch=preject_head_embed,
                            labels=y,
                            device=device,
                            temperature=args.temperature,
                            base_temperature=args.temperature)

                        step_loss = step_loss + batch_sup_contrastive_loss * args.sup_con_mu

                        batch_sup_contrastive_loss_feq = sup_contrastive_loss(
                            embd_batch=preject_head_embed_feq,
                            labels=y,
                            device=device,
                            temperature=args.temperature,
                            base_temperature=args.temperature)

                        step_loss_feq = step_loss_feq + batch_sup_contrastive_loss_feq * args.sup_con_lamda

                    step_loss.backward()
                    step_loss_feq.backward()
                    optimizer.step()
                    optimizer_feq.step()

                    num_iterations = num_iterations + 1
            else:
                ## Self-supervised contrasting
                for x, x_feq, x_aug1, x_aug2, freq_aug1, freq_aug2, y, mask_train_batch in train_loader:
                    if x.shape[0] < 2:
                        continue

                    y_feq = y.clone()

                    optimizer.zero_grad()
                    optimizer_feq.zero_grad()

                    pred_embed_aug1 = model(x_aug1)
                    pred_embed_aug2 = model(x_aug2)

                    pred_embed_feq_aug1 = model_feq(freq_aug1)
                    pred_embed_feq_aug2 = model_feq(freq_aug2)

                    if is_projection_head:
                        preject_embed_aug1 = projection_head_aug1(pred_embed_aug1)
                        preject_embed_aug2 = projection_head_aug2(pred_embed_aug2)
                        preject_embed_feq_aug1 = projection_head_feq_aug1(pred_embed_feq_aug1)
                        preject_embed_feq_aug2 = projection_head_feq_aug2(pred_embed_feq_aug2)

                    step_loss = smclr_criterion(preject_embed_aug1, preject_embed_aug2)
                    step_loss_feq = smclr_criterion_freq(preject_embed_feq_aug1, preject_embed_feq_aug2)
                    step_loss.backward()
                    step_loss_feq.backward()
                    optimizer.step()
                    optimizer_feq.step()
                    
                ## Supervised contrasting and pseudo-label learning
                for x, x_feq, x_aug1, x_aug2, freq_aug1, freq_aug2, y, mask_train_batch in train_loader:
                    if x.shape[0] < 2:
                        continue

                    y_feq = y.clone()

                    optimizer.zero_grad()
                    optimizer_feq.zero_grad()
                    
                    pred_embed = model(x)
                    pred_embed_aug1 = model(x_aug1)
                    pred_embed_aug2 = model(x_aug2)

                    pred_embed_feq = model_feq(x_feq)
                    pred_embed_feq_aug1 = model_feq(freq_aug1)
                    pred_embed_feq_aug2 = model_feq(freq_aug2)

                    if is_projection_head:
                        preject_head_embed = projection_head(pred_embed)
                        preject_head_embed_feq = projection_head_feq(pred_embed_feq)

                    mask_cpl_batch = torch.tensor([False] * len(mask_train_batch)).to(device)
                    mask_cpl_batch_feq = torch.tensor([False] * len(mask_train_batch)).to(device)

                    if epoch >= args.warmup_epochs:

                        if not queue_train_x.full():
                            queue_train_x.put(preject_head_embed.detach())
                            queue_train_y.put(y)
                            queue_train_mask.put(mask_train_batch)

                            queue_train_x_feq.put(preject_head_embed_feq.detach())
                            queue_train_y_feq.put(y)

                        if queue_train_x.full():
                            train_x_allq = queue_train_x.queue
                            train_y_allq = queue_train_y.queue
                            train_mask_allq = queue_train_mask.queue

                            train_x_allq_feq = queue_train_x_feq.queue
                            train_y_allq_feq = queue_train_y_feq.queue

                            embed_train_x_allq = torch.cat([train_x_allq[j] for j in range(len(train_x_allq))], 0)
                            y_label_allq = torch.cat([train_y_allq[j] for j in range(len(train_y_allq))], 0)
                            mask_lable_allq = np.concatenate(train_mask_allq)

                            embed_train_x_allq_feq = torch.cat(
                                [train_x_allq_feq[j] for j in range(len(train_x_allq_feq))], 0)
                            y_label_allq_feq = torch.cat([train_y_allq_feq[j] for j in range(len(train_y_allq_feq))], 0)

                            _, end_knn_label, mask_cpl_knn, _ = construct_graph_via_knn_cpl_nearind_gpu(
                                data_embed=embed_train_x_allq, y_label=y_label_allq,
                                mask_label=mask_lable_allq, device=device,
                                num_real_class=args.num_classes, topk=args.knn_num_tem)
                            knn_result_label = torch.tensor(end_knn_label).to(device).to(torch.int64)
                            y[mask_train_batch == 1] = knn_result_label[(len(knn_result_label) - len(y)):][
                                mask_train_batch == 1]
                            mask_cpl_batch[mask_train_batch == 1] = mask_cpl_knn[(len(mask_cpl_knn) - len(y)):][
                                mask_train_batch == 1]

                            _, end_knn_label_feq, mask_cpl_knn_feq, _ = construct_graph_via_knn_cpl_nearind_gpu(
                                data_embed=embed_train_x_allq_feq, y_label=y_label_allq_feq,
                                mask_label=mask_lable_allq, device=device,
                                num_real_class=args.num_classes, topk=args.knn_num_feq)
                            knn_result_label_feq = torch.tensor(end_knn_label_feq).to(device).to(torch.int64)
                            y_feq[mask_train_batch == 1] = knn_result_label_feq[(len(knn_result_label_feq) - len(y)):][
                                mask_train_batch == 1]
                            mask_cpl_batch_feq[mask_train_batch == 1] = \
                                mask_cpl_knn_feq[(len(knn_result_label_feq) - len(y)):][mask_train_batch == 1]

                            _ = queue_train_x.get()
                            _ = queue_train_y.get()
                            _ = queue_train_mask.get()

                            _ = queue_train_x_feq.get()
                            _ = queue_train_y_feq.get()

                    mask_clean = [True if mask_train_batch[m] == 0 else False for m in range(len(mask_train_batch))]
                    mask_select_loss = [False for m in range(len(y))]
                    mask_select_loss_feq = [False for m in range(len(y))]
                    for m in range(len(mask_train_batch)):
                        if mask_train_batch[m] == 0:
                            mask_select_loss[m] = True
                            mask_select_loss_feq[m] = True
                        else:
                            if mask_cpl_batch[m]:
                                mask_select_loss[m] = True
                            if mask_cpl_batch_feq[m]:
                                mask_select_loss_feq[m] = True

                    pred = classifier(pred_embed)
                    pred_aug1 = classifier(pred_embed_aug1)
                    pred_aug2 = classifier(pred_embed_aug2)

                    pred_feq = classifier_feq(pred_embed_feq)
                    pred_feq_aug1 = classifier_feq(pred_embed_feq_aug1)
                    pred_feq_aug2 = classifier_feq(pred_embed_feq_aug2)

                    hyp_epoch = 0.5
                    if len(y_feq[mask_train_batch == 0]) > 1:

                        step_loss = loss(pred[mask_select_loss_feq], y_feq[mask_select_loss_feq]) \
                                    + hyp_epoch * (
                                            loss(pred_aug1[mask_train_batch == 0], y_feq[mask_train_batch == 0]) +
                                            loss(pred_aug2[mask_train_batch == 0], y_feq[mask_train_batch == 0]))

                        step_loss_feq = loss_feq(pred_feq[mask_select_loss], y[mask_select_loss]) + \
                                        hyp_epoch * (loss_feq(pred_feq_aug1[mask_train_batch == 0],
                                                              y[mask_train_batch == 0]) +
                                                     loss_feq(pred_feq_aug2[mask_train_batch == 0],
                                                              y[mask_train_batch == 0]))
                    else:
                        step_loss = loss(pred[mask_select_loss_feq], y_feq[mask_select_loss_feq])

                        step_loss_feq = loss_feq(pred_feq[mask_select_loss], y[mask_select_loss])

                    if epoch > args.warmup_epochs:

                        if len(y[mask_train_batch == 0]) > 1:
                            batch_sup_contrastive_loss = sup_contrastive_loss(
                                embd_batch=preject_head_embed[mask_train_batch == 0],
                                labels=y[mask_train_batch == 0],
                                device=device,
                                temperature=args.temperature,
                                base_temperature=args.temperature)

                            step_loss = step_loss + batch_sup_contrastive_loss * args.sup_con_mu

                            batch_sup_contrastive_loss_feq = sup_contrastive_loss(
                                embd_batch=preject_head_embed_feq[mask_train_batch == 0],
                                labels=y_feq[mask_train_batch == 0],
                                device=device,
                                temperature=args.temperature,
                                base_temperature=args.temperature)

                            step_loss_feq = step_loss_feq + batch_sup_contrastive_loss_feq * args.sup_con_lamda
                    
                    reg_loss = reg_co_training_loss(time_pre=pred, freq_pre=pred_feq.detach()).to(device)
                    
                    reg_loss_freq = reg_co_training_loss(time_pre=pred.detach(), freq_pre=pred_feq).to(device)

                    step_loss = step_loss + reg_loss
                    step_loss_feq = step_loss_feq + reg_loss_freq

                    step_loss.backward()
                    step_loss_feq.backward()
                    optimizer.step()
                    optimizer_feq.step()

                    num_iterations += 1

            model.eval()
            classifier.eval()
            projection_head.eval()
            projection_head_aug1.eval()
            projection_head_aug2.eval()
            
            model_feq.eval()
            classifier_feq.eval()
            projection_head_feq.eval()
            projection_head_feq_aug1.eval()
            projection_head_feq_aug2.eval()

            val_loss, val_accu_tem = evaluate(val_loader, model, classifier, loss, device)
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                end_val_epoch = epoch
                test_loss, test_accuracy_tem = evaluate(test_loader, model, classifier, loss, device)

            val_loss_feq, val_accu_feq = evaluate(val_loader_feq, model_feq, classifier_feq, loss_feq, device)
            if min_val_loss_feq > val_loss_feq:
                min_val_loss_feq = val_loss_feq
                end_val_epoch_feq = epoch
                test_loss_feq, test_accuracy_feq = evaluate(test_loader_feq, model_feq, classifier_feq,
                                                            loss_feq, device)

            if val_accu_tem >= val_accu_feq:  ## The end test accuracy
                test_accuracy = test_accuracy_tem
            else:
                test_accuracy = test_accuracy_feq

            if (epoch > args.warmup_epochs) and (abs(last_loss - val_loss) <= 1e-4):
                stop_count += 1
            else:
                stop_count = 0

            if (epoch > args.warmup_epochs) and (val_loss > last_loss):
                increase_count += 1
            else:
                increase_count = 0

            last_loss = val_loss

            if epoch % 10 == 0:
                print("epoch : {},  test_accuracy : {:.3f}".format(epoch, test_accuracy))

        test_accuracies.append(test_accuracy)
        t = time.time() - t
        train_time += t

        print("Finish training")

    test_accuracies = torch.Tensor(test_accuracies)
    print("Training end: mean_test_acc = ", round(torch.mean(test_accuracies).item(), 3), "traning time (seconds) = ",
          round(train_time, 3), ", seed = ", args.random_seed)

    print('Done!')
