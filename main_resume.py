import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from utils.model_utils import load_model
from utils.index_selection import get_indices_per_subfolder
from utils.initial_subset import get_initial_subset
from utils.get_uncertainty_score import get_uncertainty_scores
from utils.train import train_model
from utils.train_clf import train_clf
from utils.cifar10_selection import get_indices_cifar10
from net.resnet18 import resnet18
from net.resnet18_cifar import resnet18_cifar
from net.resnet50 import resnet50
from config import get_config
Config = get_config()
if Config.DATASET == 'VGGDataset':
    from data.VGGdataset import get_transforms, load_datasets
elif Config.DATASET == 'Mingren':
    from data.Mingren import get_transforms, load_datasets
elif Config.DATASET == 'CelebA':
    from data.CelebA import get_transforms, load_datasets
elif Config.DATASET == 'Cifar10':
    from data.Cifar10 import get_transforms, load_datasets

torch.backends.cudnn.deterministic = True
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    for attr, value in vars(Config).items():
        print(f"{attr}: {value}")
    set_seed(123)    
    device = Config.DEVICE
    run_id = 1
    while True:
        active_learning_dir = os.path.join(Config.MODEL_SAVE_DIR, f"Active_Learning_{run_id}")
        
        if not os.path.exists(active_learning_dir):
            os.makedirs(active_learning_dir)
            break
        run_id += 1
        print(active_learning_dir)
    train_transform, test_transform = get_transforms()

    full_dataset, test_dataset = load_datasets(train_transform, test_transform)
    if Config.DATASET =='Cifar10':
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        selected_indices = indices[:Config.INITIAL_SUBSET_NUMBER]
        print(f"Number of initially selected images: {len(selected_indices)}")
        unlabeled_indices = indices[Config.INITIAL_SUBSET_NUMBER:]
        print(f"Number of unlabeled images: {len(unlabeled_indices)}")
        dst_train = Subset(full_dataset, selected_indices)
        dst_unlabeled = Subset(full_dataset, unlabeled_indices)
    else:
        selected_indices = get_initial_subset(img_root = Config.IMG_ROOT_TRAIN, n = Config.INITIAL_SUBSET_NUMBER, dataset = full_dataset )   
        print(f"Number of initially selected images: {len(selected_indices)}")

        all_indices = set(range(len(full_dataset)))
        unlabeled_indices = list(all_indices - set(selected_indices))
        print(f"Number of unlabeled images: {len(unlabeled_indices)}")

        # Create Subset
        dst_train = Subset(full_dataset, selected_indices)
        dst_unlabeled = Subset(full_dataset, unlabeled_indices)
    
    for cycle in range(1, Config.ACTIVE_LEARNING_CYCLES + 1):
        print(f"\n===== Active Learning Cycle {cycle} =====")
        

        current_model_path = os.path.join(active_learning_dir, f"cycle_{cycle}.pth")

        previous_model_path = os.path.join(active_learning_dir, f"cycle_{cycle-1}.pth")

        if Config.DATASET == 'VGGDataset':
            model = resnet18(num_classes=47).to(device)
        elif Config.DATASET == 'Mingren':
            model = resnet18(num_classes=10).to(device)
        elif Config.DATASET == 'CelebA':
            model = resnet18(num_classes=40).to(device)
        elif Config.DATASET == 'Cifar10':
            model = resnet18_cifar(num_classes=10).to(device)

        if cycle == 1:
            model = model
            num_epochs = Config.NUM_EPOCHS
        
        else:
            if Config.CPT == 'False':
                model == model
            else:
                model = load_model(model, previous_model_path, device)
                num_epochs = Config.NUM_EPOCHS
                print(f'\n===== LOAD MODEL {previous_model_path} =====')

        train_dataloader = DataLoader(
            dst_train, 
            batch_size=Config.BATCH_SIZE, 
            num_workers=Config.NUM_WORKERS, 
            shuffle=True, 
            pin_memory=True
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=Config.BATCH_SIZE, 
            num_workers=Config.NUM_WORKERS, 
            shuffle=False
        )
        
        if Config.DATASET == 'Cifar10':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov = True)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

        if Config.DATASET in ['VGGDataset', 'Mingren', 'CelebA']:
            criterion = torch.nn.BCEWithLogitsLoss()
            train_model(
                train_dataloader=train_dataloader, 
                model=model, 
                optimizer=optimizer, 
                criterion=criterion, 
                test_dataloader=test_dataloader,
                test_interval=Config.TEST_INTERVAL, 
                num_epochs=num_epochs, 
                save_best_model=True, 
                best_f1=0.0, 
                best_val_accuracy=0.0,
                save_path = current_model_path,
                cycle = cycle
                    )
        elif Config.DATASET == 'Cifar10': 
            criterion = torch.nn.CrossEntropyLoss()
            train_clf(
                train_dataloader=train_dataloader, 
                model=model, 
                optimizer=optimizer, 
                criterion=criterion, 
                test_dataloader=test_dataloader,
                test_interval=Config.TEST_INTERVAL, 
                num_epochs=num_epochs, 
                save_best_model=True, 
                best_val_accuracy=0.0,
                save_path = current_model_path,
                cycle = cycle
                    )
        else:
            raise ValueError(f"Unsupported dataset: {Config.DATASET}")

        
        if cycle < Config.ACTIVE_LEARNING_CYCLES:
            #caculate uncertainty score             
            scores = get_uncertainty_scores(model, dst_unlabeled, dst_train, device)     

            index_to_score = {full_idx: score for full_idx, score in zip(unlabeled_indices, scores)}
            if  Config.DATASET == 'Cifar10':
                new_selected_indices = get_indices_cifar10(dst_unlabeled, n=Config.N_SELECTED_IMAGES_PER_CYCLE, uncertainty_scores=index_to_score)
            else:
                new_selected_indices = get_indices_per_subfolder(
                    img_root=Config.IMG_ROOT_TRAIN, 
                    n=Config.N_SELECTED_IMAGES_PER_CYCLE, 
                    dataset=dst_unlabeled, 
                    uncertainty_scores=index_to_score,
                    blur_threshold=Config.BLUR_THRESHOLD
                )

            print(f"Selected indices for Cycle {cycle}: {new_selected_indices}")
            

            selected_indices += new_selected_indices
            dst_train = Subset(full_dataset, selected_indices)
            unlabeled_indices = list(set(unlabeled_indices) - set(new_selected_indices))


            print(f"Number of selected images for Cycle {cycle}: {len(selected_indices)}")
            print(f"Number of unlabeled images after Cycle {cycle}: {len(unlabeled_indices)}")
            dst_unlabeled = Subset(full_dataset, unlabeled_indices)
            torch.cuda.empty_cache()
            print(f"\n===== Active Learning Cycle {cycle} Done !=====")
            
        else:    
            print("\n===== Your Active Learning Process is Done ! =====")

if __name__ == "__main__":
    main()