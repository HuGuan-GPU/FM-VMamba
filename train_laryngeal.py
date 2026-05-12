import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import torch
torch.cuda.empty_cache()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


try:
    from vmamba import VSSM 
    print("成功导入官方 VMamba 模型！")
except ImportError as e:
    print(f"导入失败，请检查目录结构或编译环境。错误信息: {e}")
    sys.exit(1)


CONFIG = {
    'data_dir': '/root/FM-Vmamba/Vmamba/OriginalData',
    'batch_size': 4,
    'lr': 1e-4,
    'num_epochs': 50,
    'num_classes': 6,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'val_split': 0.2,
    'save_model_path': 'best_vmamba_laryngeal.pth'
}


train_transform = transforms.Compose([
    # transforms.Resize((256, 256))
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    import shutil
    hidden_dir = os.path.join(CONFIG['data_dir'], '.ipynb_checkpoints')
    if os.path.exists(hidden_dir):
        shutil.rmtree(hidden_dir)
        print("已自动揪出并删除了隐藏的.ipynb_checkpoints文件夹！")
    dataset = datasets.ImageFolder(CONFIG['data_dir'], transform=train_transform)
    print(f"检测到类别: {dataset.class_to_idx}")
    
    num_val = int(len(dataset) * CONFIG['val_split'])
    num_train = len(dataset) - num_val
    train_set, val_set = random_split(dataset, [num_train, num_val])
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    model = VSSM(
        patch_size=4,
        in_chans=3,
        num_classes=CONFIG['num_classes'], 
        depths=[2, 2, 9, 2],
        dims=96,
        drop_rate=0.0,
        drop_path_rate=0.1,
    ).to(CONFIG['device'])
    
    # model = VSSM(
    #     patch_size=4,
    #     in_chans=3,
    #     num_classes=CONFIG['num_classes'], 
    #     depths=[1, 1, 3, 1],
    #     dims=24,
    #     drop_rate=0.0,
    #     drop_path_rate=0.1,
    # ).to(CONFIG['device'])

    if torch.cuda.device_count() > 1:
        print(f"-> 检测到多卡，正在使用 {torch.cuda.device_count()} 张 GPU 进行 DataParallel 并行训练！")
        model = nn.DataParallel(model)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
    
    # # 4.3 训练循环
    # best_acc = 0.0
    # for epoch in range(CONFIG['num_epochs']):
    #     model.train()
    #     running_loss = 0.0
    #     correct = 0
    #     total = 0
        
    #     train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Train]")
    #     for images, labels in train_bar:
    #         images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
            
    #         running_loss += loss.item() * images.size(0)
    #         _, predicted = torch.max(outputs, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
    #     train_acc = 100 * correct / total
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)

    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        train_acc = 100 * correct / total
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Val]"):
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # # 4.5 保存权重
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     torch.save(model.state_dict(), CONFIG['save_model_path'])
        #     print("-> 已保存最佳模型权重")

        if val_acc > best_acc:
            best_acc = val_acc
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), CONFIG['save_model_path'])
            else:
                torch.save(model.state_dict(), CONFIG['save_model_path'])
            print("-> 已保存最佳模型权重")

        torch.cuda.empty_cache()
if __name__ == "__main__":
    main()