import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from utils.test_clf import test  # 确保 test 函数适用于多分类并返回准确率
from config import get_config

Config = get_config()
device = Config.DEVICE

def train_clf(
    train_dataloader,
    model,
    optimizer,
    criterion,
    test_dataloader=None,
    test_interval=10,
    num_epochs=100,
    save_best_model=True,
    best_val_accuracy=0.0,
    save_path='/path/to/save/best_model.pth',
    cycle=0
):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.NUM_EPOCHS,
        eta_min=1e-4
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        for img, labels in train_dataloader:
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(img)  # 输出通常为 [batch_size, num_classes]
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # 获取预测类别
            _, predicted = torch.max(output, 1)
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        train_accuracy = accuracy_score(all_labels, all_predictions) * 100
        running_loss = running_loss / len(train_dataloader)
        
        print(f'Train Epoch {epoch + 1}, Loss: {running_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        scheduler.step()
        
        # 每隔 test_interval 进行测试
        if test_dataloader is not None and (epoch + 1) % test_interval == 0:
            test_accuracy = test(model, test_dataloader)  # 假设 test 函数返回准确率
            print(f'--- Test after Epoch {epoch + 1} ---')
            print(f'Test Accuracy: {test_accuracy:.2f}%')
            
            # 保存最佳模型
            if Config.SAVE == True:
                if save_best_model and test_accuracy > best_val_accuracy:
                    best_val_accuracy = test_accuracy
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': running_loss,
                    }, save_path)
                    print(f"Cycle {cycle} Model saved at epoch {epoch + 1} with Accuracy {test_accuracy:.2f}%")

    
    print("Training complete.")
