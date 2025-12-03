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

from ts_data.dataloader import UCRDataset, UCRTFAugDataset
from ts_data.preprocessing import normalize_per_series, fill_nan_value
from ts_model.loss import sup_contrastive_loss, reg_co_training_loss, SimCLRContrastiveLoss
from ts_model.model import ProjectionHead
from utils.ts_utils import build_model, set_seed, build_dataset, get_all_datasets, \
    construct_graph_via_knn_cpl_nearind_gpu, \
    build_loss, shuffler, evaluate, convert_coeff
from utils.aug_utils import get_freq_augmentation, get_time_augmentation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Base setup
    parser.add_argument('--backbone', type=str, default='fcn', help='encoder backbone, fcn')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='CBF', help='dataset(in ucr)') 
    parser.add_argument('--dataroot', type=str, default='./UCRArchive_2018', help='path of UCR folder')
    parser.add_argument('--num_classes', type=int, default=0, help='number of class')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')

    # Semi training
    parser.add_argument('--labeled_ratio', type=float, default=0.1, help='0.1, 0.2, 0.4')
    parser.add_argument('--warmup_epochs', type=int, default=300, help='warmup epochs using only labeled data for ssc')
    parser.add_argument('--queue_maxsize', type=int, default=3, help='2 or 3')
    parser.add_argument('--knn_num_tem', type=int, default=40, help='10, 20, 50')
    parser.add_argument('--knn_num_feq', type=int, default=30, help='10, 20, 50')

    ## Augmentation
    parser.add_argument('--aug_time_method_index', type=int, default=1, help='1,2,3,4,5,6')
    parser.add_argument('--aug2_time_method_index', type=int, default=5, help='1,2,3,4,5,6')
    parser.add_argument('--aug_freq_method_index', type=int, default=1, help='1,2,3,4')
    parser.add_argument('--aug2_freq_method_index', type=int, default=3, help='1,2,3,4')
    parser.add_argument('--self_contra_tau', type=float, default=0.1, help='0.1') 

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
    parser.add_argument('--epoch', type=int, default=1000, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    # classifier setup
    parser.add_argument('--classifier', type=str, default='linear', help='')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)

    sum_dataset, sum_target, num_classes = build_dataset(args)
    args.num_classes = num_classes
    args.seq_len = sum_dataset.shape[1]

    while sum_dataset.shape[0] * 0.6 < args.batch_size:
        args.batch_size = args.batch_size // 2

    if args.batch_size * 2 > sum_dataset.shape[0] * 0.6:
        args.queue_maxsize = 2
        
    ## Time Domain Settings
    model, classifier = build_model(args)
    projection_head = ProjectionHead()
    projection_head_aug1 = ProjectionHead()
    projection_head_aug2 = ProjectionHead()

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
    args.input_size = 2
    model_feq, classifier_feq = build_model(args)
    projection_head_feq = ProjectionHead()
    projection_head_feq_aug1 = ProjectionHead()
    projection_head_feq_aug2 = ProjectionHead()

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

    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(sum_dataset, sum_target)

    losses = []
    test_accuracies = []
    train_time = 0.0
    end_val_epochs = []

    test_accuracies_tem = []
    end_val_epochs_tem = []
    test_accuracies_feq = []
    end_val_epochs_feq = []

    for i, train_dataset in enumerate(train_datasets):
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

        print('{} fold start training and evaluate'.format(i))

        train_target = train_targets[i]
        val_dataset = val_datasets[i]
        val_target = val_targets[i]

        test_dataset = test_datasets[i]
        test_target = test_targets[i]

        train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)

        # TODO normalize per series
        train_dataset = normalize_per_series(train_dataset)
        val_dataset = normalize_per_series(val_dataset)
        test_dataset = normalize_per_series(test_dataset)

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

        aug_x_train_all = get_time_augmentation(train_dataset=x_train_all,
                                                method_index=args.aug_time_method_index)
        aug_x_train_all = torch.from_numpy(aug_x_train_all).float().to(device)
        aug2_x_train_all = get_time_augmentation(train_dataset=x_train_all,
                                                 method_index=args.aug2_time_method_index)
        aug2_x_train_all = torch.from_numpy(aug2_x_train_all).float().to(device)

        train_fft = fft.rfft(torch.from_numpy(x_train_all), dim=-1)
        train_fft, _ = convert_coeff(train_fft)
        train_fft = train_fft.to(device)

        aug_x_freq_train_all = get_freq_augmentation(torch_train_dataset=torch.from_numpy(x_train_all),
                                                     method_index=args.aug_freq_method_index)

        if args.aug_freq_method_index != 4:
            aug_x_freq_train_all = fft.rfft(aug_x_freq_train_all, dim=-1)
            aug_x_freq_train_all, _ = convert_coeff(aug_x_freq_train_all)
        aug_x_freq_train_all = aug_x_freq_train_all.to(device)

        aug2_x_freq_train_all = get_freq_augmentation(torch_train_dataset=torch.from_numpy(x_train_all),
                                                      method_index=args.aug2_freq_method_index)
        if args.aug2_freq_method_index != 4:
            aug2_x_freq_train_all = fft.rfft(aug2_x_freq_train_all, dim=-1)
            aug2_x_freq_train_all, _ = convert_coeff(aug2_x_freq_train_all)
        aug2_x_freq_train_all = aug2_x_freq_train_all.to(device)

        x_train_all = torch.from_numpy(x_train_all).to(device)
        y_train_all = torch.from_numpy(y_train_all).to(device).to(torch.int64)

        x_train_labeled_all = torch.unsqueeze(x_train_all[mask_train == 0], 1)
        x_train_labeled_all_feq = train_fft[mask_train == 0]
        y_train_labeled_all = y_train_all[mask_train == 0]

        x_train_aug_labeled_all = torch.unsqueeze(aug_x_train_all[mask_train == 0], 1)
        x_train_aug2_labeled_all = torch.unsqueeze(aug2_x_train_all[mask_train == 0], 1)

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

        val_set = UCRDataset(torch.from_numpy(val_dataset).to(device),
                             torch.from_numpy(val_target).to(device).to(torch.int64))
        test_set = UCRDataset(torch.from_numpy(test_dataset).to(device),
                              torch.from_numpy(test_target).to(device).to(torch.int64))

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

        val_fft = fft.rfft(torch.from_numpy(val_dataset), dim=-1)
        val_fft, _ = convert_coeff(val_fft)
        val_set_feq = UCRDataset(val_fft.to(device), torch.from_numpy(val_target).to(device).to(torch.int64))

        test_fft = fft.rfft(torch.from_numpy(test_dataset), dim=-1)
        test_fft, _ = convert_coeff(test_fft)
        test_set_feq = UCRDataset(test_fft.to(device), torch.from_numpy(test_target).to(device).to(torch.int64))

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

            if stop_count == 80 or increase_count == 80:
                print('model convergent at epoch {}, early stopping'.format(epoch))
                break

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

            if epoch % 100 == 0:
                print("epoch : {},  test_accuracy : {:.3f}".format(epoch, test_accuracy))

        test_accuracies.append(test_accuracy)   
        t = time.time() - t
        train_time += t

        print('{} fold finish training'.format(i))

    test_accuracies = torch.Tensor(test_accuracies)

    print("Training end: mean_test_acc = ", round(torch.mean(test_accuracies).item(), 3), "traning time (seconds) = ",
          round(train_time, 3), ", seed = ", args.random_seed)

    print('Done!')
