# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from utils.test import test
from config import get_config
Config = get_config()
device = Config.DEVICE

def train_model(train_dataloader, model, optimizer, criterion, test_dataloader=None, test_interval=10, num_epochs=100, save_best_model=True, best_f1=0.0, best_val_accuracy=0.0, save_path='/path/to/save/best_model.pth', cycle = 0):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for img, labels in train_dataloader:
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (torch.sigmoid(output) > 0.5).float()
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)  
        all_predictions = np.concatenate(all_predictions, axis=0)

        weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')

        total = all_labels.size
        correct = (all_labels == all_predictions).sum()
        train_accuracy  = 100.0 * correct / total
        running_loss = running_loss / len(train_dataloader)
        
        print(f'Train Epoch {epoch + 1}, Loss: {running_loss:.4f}, Accuracy: {train_accuracy:.2f}%, weighted_f1: {weighted_f1:.2f}')
        scheduler.step(running_loss)
        
        # test every test_interval
        if test_dataloader is not None and (epoch + 1) % test_interval == 0:
            test_accuracy, test_f1, mAP, balanced_acc  = test(model, test_dataloader)
            print(f'--- Cycle {cycle} Test after Epoch {epoch + 1} ---')
            print(f'Test Accuracy: {test_accuracy:.2f}%, Test Weighted F1-score: {test_f1:.4f}')
            print(f'mAP: {mAP:.4f}')
            print(f'Balanced Accuracy: {balanced_acc:.4f}')   
            # save best model
            if Config.SAVE == True:
                if save_best_model and test_f1 > best_f1:
                    best_f1 = test_f1
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': running_loss,
                    }, save_path)
                    print(f"Cycle {cycle} Model saved at epoch {epoch + 1} with F1-score {test_f1:.4f}")
    print("Training complete.")
