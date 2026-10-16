import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
import random
import os
import time
import gc



from models.CSSTNet import Mymodel



DATA_DIR = os.getcwd() + "/lyx_WorkSpace/Datasets/Mydata2/"
CHECKPOINT_DIR = os.getcwd() + "/lyx_WorkSpace/pytorch/checkpoint/"
RESULT_DIR = os.getcwd() + "/lyx_WorkSpace/pytorch/result/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



SEED = 42
BATCHSIZE = 72
LR = 0.001
EPOCHS = 1000
CLASS_NUM = 4
START_NUM = 1
END_NUM = 9
K_FOLD_NUM = 10
VALIDATE_RATIO = 0.3



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# ==============================
# SR 
# ==============================
def interaug_gpu(timg, label, batch_size, class_num, number_augmentation=4, number_seg=8):
    N, _, C, T = timg.shape
    seg_len = T // number_seg
    device = timg.device
    
    unique_labels = torch.unique(label)
    aug_data_list = []
    aug_label_list = []
    
    for cls in unique_labels:
        cls_mask = (label == cls)
        cls_data = timg[cls_mask]  # [n_cls, 1, C, T]
        n_cls = cls_data.size(0)
        
        if n_cls == 0:
            continue
        
        num_aug = number_augmentation * int(batch_size / class_num)
        
        aug_cls_data = torch.zeros((num_aug, 1, C, T), dtype=timg.dtype, device=device)
        
        for rj in range(number_seg):
            rand_indices = torch.randint(0, n_cls, (num_aug,), device=device)
            
            start = rj * seg_len
            end = (rj + 1) * seg_len
            
            aug_cls_data[:, :, :, start:end] = cls_data[rand_indices, :, :, start:end]
            
        aug_data_list.append(aug_cls_data)
        aug_label_list.append(torch.full((num_aug,), cls, dtype=torch.long, device=device))
        
    if len(aug_data_list) > 0:
        aug_data = torch.cat(aug_data_list, dim=0)
        aug_label = torch.cat(aug_label_list, dim=0)
        
        shuffle_idx = torch.randperm(aug_data.size(0), device=device)
        aug_data = aug_data[shuffle_idx]
        aug_label = aug_label[shuffle_idx]
        
        return aug_data, aug_label
    else:
        return torch.empty((0, 1, C, T), device=device), torch.empty((0,), device=device)



# ==============================
# train
# ==============================
def train_pytorch(train_x, train_y, test_x, test_y, subject_id, fold, device):

    model = Mymodel()
    modelname = 'CSSTNet'
    
    model = model.to(device)
    
    print('----------------------%s----------------------' % (modelname))
    print(model)
    print("----------------subject:%s fold:%s--------------------" % (subject_id, fold))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    train_x_tensor = torch.from_numpy(train_x).float()
    train_y_tensor = torch.from_numpy(train_y).long()
    test_x_tensor = torch.from_numpy(test_x).float()
    test_y_tensor = torch.from_numpy(test_y).long()
    
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    
    train_x_tensor_gpu = train_x_tensor.to(device)
    train_y_tensor_gpu = train_y_tensor.to(device)
    
    check_file = "best_model_A{:02d}_fold_{:02d}.pth".format(subject_id, fold)
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, modelname, "BCI4_2a", "CV")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, check_file)
    
    best_val_loss = float('inf')
    
    epoch_pbar = tqdm(range(EPOCHS), desc=f"Sub{subject_id} Fold{fold}", unit="epoch")
    
    for epoch in epoch_pbar:
        ''' ---------- train ---------- '''
        train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        val_data_list = []
        val_label_list = []
        
        for inputs, labels in train_loader:
            number_sample = inputs.shape[0]
            number_validate = int(VALIDATE_RATIO * number_sample)
            
            train_inputs_part = inputs[:-number_validate]
            train_labels_part = labels[:-number_validate]
            
            val_inputs_part = inputs[-number_validate:]
            val_labels_part = labels[-number_validate:]
            
            val_data_list.append(val_inputs_part)
            val_label_list.append(val_labels_part)
            
            train_inputs_part = train_inputs_part.to(device)
            train_labels_part = train_labels_part.to(device)
            
            aug_inputs, aug_labels = interaug_gpu(
                train_x_tensor_gpu, 
                train_y_tensor_gpu, 
                batch_size=BATCHSIZE, 
                class_num=CLASS_NUM
            )
            
            inputs_final = torch.cat((train_inputs_part, aug_inputs))
            labels_final = torch.cat((train_labels_part, aug_labels))
            
            optimizer.zero_grad()
            outputs = model(inputs_final)
            loss = criterion(outputs, labels_final)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs_final.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels_final.size(0)
            correct_train += (predicted == labels_final).sum().item()
        
        train_loss = running_loss / total_train
        train_acc = 100.0 * correct_train / total_train
        
        ''' ---------- validation ---------- '''
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_inputs_all = torch.cat(val_data_list)
        val_labels_all = torch.cat(val_label_list)
        
        val_dataset = TensorDataset(val_inputs_all, val_labels_all)
        val_loader_dynamic = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False)
        
        with torch.no_grad():
            for inputs, labels in val_loader_dynamic:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss /= total_val
        val_acc = 100.0 * correct_val / total_val
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
        
        epoch_pbar.set_postfix({
            'TAcc': f'{train_acc:.2f}%',
            'TLoss': f'{train_loss:.4f}',
            'Vacc': f'{val_acc:.2f}%',
            'Vloss': f'{val_loss:.4f}',
            'Minloss': f'{best_val_loss:.4f}'
        })
    
    ''' -------- test --------- '''
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    test_x_tensor = test_x_tensor.to(device)
    test_y_tensor = test_y_tensor.to(device)
    
    with torch.no_grad():
        test_outputs = model(test_x_tensor)
        _, preds = torch.max(test_outputs, 1)

        test_acc = round(100.0 * (preds.cpu() == test_y_tensor.cpu()).float().mean().item(), 2)
        kappa = cohen_kappa_score(test_y, preds.cpu().numpy())
    
    print("Fold {} Test Accuracy: {:.2f}%, Kappa: {:.4f}".format(fold, test_acc, kappa))
    
    return test_acc, modelname, kappa


# ==============================
# main
# ==============================
if __name__ == '__main__':
    set_seed(SEED)
    device = DEVICE
    print(f"Using device: {device}")
    
    alg_df = pd.DataFrame(columns=['Test acc', 'Kappa'])
    alg_record = []
    kappa_record = []
    
    starttime_str = time.strftime('%Y_%m_%d_%H-%M-%S', time.localtime())
    start_time = time.time()
    
    for subject_id in range(START_NUM, END_NUM + 1):
        mat_data = sio.loadmat(DATA_DIR + 'A0%d.mat' % (subject_id))
        
        train_x = mat_data['train_data']
        test_x = mat_data['test_data']
        train_y = mat_data['train_labels']
        test_y = mat_data['test_labels']
        
        train_x = np.expand_dims(train_x, axis=1)
        test_x = np.expand_dims(test_x, axis=1)
        
        train_y = (train_y - 1).squeeze()
        test_y = (test_y - 1).squeeze()
        
        train_mean = np.mean(train_x)
        train_std = np.std(train_x)
        train_x = (train_x - train_mean) / train_std
        test_x = (test_x - train_mean) / train_std
        
        print("=" * 60)
        print("Subject A{:02d} - train_x shape: {}, test_x shape: {}".format(
            subject_id, train_x.shape, test_x.shape))
        print("Subject A{:02d} - train_y shape: {}, test_y shape: {}".format(
            subject_id, train_y.shape, test_y.shape))
        print("=" * 60)
        

        data = np.concatenate((train_x, test_x), axis=0)  # (N_total, 1, Chans, Samples)
        labels_raw = np.concatenate((train_y, test_y), axis=0).squeeze()  # (N_total,)
        
        kfold = StratifiedKFold(n_splits=K_FOLD_NUM, shuffle=True, random_state=SEED)
        
        fold = 1
        subacc = []
        subkappa = []
        
        for train_idx, test_idx in kfold.split(data, labels_raw):
            print('\n================== Subject A{:02d} - Fold {:02d}/{} =================='.format(
                subject_id, fold, K_FOLD_NUM))
            
            train_x_fold = data[train_idx]
            test_x_fold = data[test_idx]
            train_y_fold = labels_raw[train_idx]
            test_y_fold = labels_raw[test_idx]
            
            test_acc, modelname, kappa = train_pytorch(
                train_x=train_x_fold,
                train_y=train_y_fold,
                test_x=test_x_fold,
                test_y=test_y_fold,
                subject_id=subject_id,
                fold=fold,
                device=device
            )
            
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"{modelname} - Subject A{subject_id:02d} Fold {fold}: Acc = {test_acc:.2f}%, Kappa = {kappa:.4f}")
            
            subacc.append(test_acc)
            subkappa.append(kappa)
            
            fold += 1
        


        mean_acc = np.mean(subacc)
        mean_kappa = np.mean(subkappa)
        std_acc = np.std(subacc, ddof=1)
        std_kappa = np.std(subkappa, ddof=1)
        
        print("\n" + "=" * 60)
        print("Subject A{:02d} - Mean Acc: {:.2f}% ± {:.2f}%, Mean Kappa: {:.4f} ± {:.4f}".format(
            subject_id, mean_acc, std_acc, mean_kappa, std_kappa))
        print("=" * 60 + "\n")
        
        alg_record.append(mean_acc)
        kappa_record.append(mean_kappa)
        alg_df.loc[f"A{subject_id:02d}"] = [mean_acc, mean_kappa]
        
        save_dir = os.path.join(
            RESULT_DIR, modelname, 'BCI4-2A', 'CV', f'class{CLASS_NUM}', starttime_str + '_all'
        )
        os.makedirs(save_dir, exist_ok=True)
        alg_df_rounded = alg_df.round({'Test acc': 2, 'Kappa': 4})
        alg_df_rounded.to_csv(os.path.join(save_dir, 'individual_results.csv'), index=True)
    

    avg_acc = np.mean(alg_record)
    avg_kappa = np.mean(kappa_record)
    std_acc = np.std(alg_record, ddof=1)
    std_kappa = np.std(kappa_record, ddof=1)
    
    alg_df.loc['Avg'] = [avg_acc, avg_kappa]
    alg_df.loc['Std'] = [std_acc, std_kappa]
    
    save_dir = os.path.join(
        RESULT_DIR, modelname, 'BCI4-2A', 'CV', f'class{CLASS_NUM}', starttime_str + '_all'
    )
    os.makedirs(save_dir, exist_ok=True)
    
    alg_df_rounded = alg_df.round({'Test acc': 2, 'Kappa': 4})
    alg_df_rounded.to_csv(os.path.join(save_dir, 'individual_results.csv'), index=True)
    
    time_used = time.time() - start_time
    print("\n" + "=" * 60)
    print("Final Results:")
    print("Average Accuracy: {:.2f}% ± {:.2f}%".format(avg_acc, std_acc))
    print("Average Kappa: {:.4f} ± {:.4f}".format(avg_kappa, std_kappa))
    print("Total Time: {:.2f} seconds".format(time_used))
    print("=" * 60)