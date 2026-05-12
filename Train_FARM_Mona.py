import os
import sys
import shutil
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
import math
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler 
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from collections import Counter 


torch.cuda.empty_cache()
cudnn.benchmark = True 
cudnn.deterministic = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


try:
    from vmamba import VSSM 
    print("成功导入官方 VMamba 模型！")
except ImportError as e:
    print(f"导入失败，请检查目录结构或编译环境。错误信息: {e}")
    sys.exit(1)


class FARM(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.freq_controller = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * 2), 
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, groups=in_channels * 2, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid() 
        )
        self.alpha = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x):
        fft_x = torch.fft.fft2(x, norm="ortho")
        fft_x_shifted = torch.fft.fftshift(fft_x) 
        
        amplitude = torch.log1p(torch.abs(fft_x_shifted))

        M_freq = self.freq_controller(amplitude)
        fft_x_filtered = fft_x_shifted * M_freq

        fft_x_ishifted = torch.fft.ifftshift(fft_x_filtered)
        x_restored = torch.fft.ifft2(fft_x_ishifted, norm="ortho").real

        return x + self.alpha * x_restored 

class MonaAdapter(nn.Module):
    def __init__(self, dim, m=0.9):
        super().__init__()
        self.dim = dim
        self.m = m 
        
        r = max(16, dim // 8)  
        
        self.down = nn.Linear(dim, r, bias=False)
        self.act = nn.GELU() 
        self.fir = nn.Conv2d(r, r, kernel_size=3, padding=1, groups=r, bias=False)
        self.up = nn.Linear(r, dim, bias=False)
        
        nn.init.zeros_(self.up.weight)
        
        self.register_buffer('mu', torch.zeros(1, 1, 1, r))
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        is_channel_first = (x.shape[1] == self.dim) and (len(x.shape) == 4)
        x_spatial = x.permute(0, 2, 3, 1) if is_channel_first else x

        feat = self.down(x_spatial)
        feat = self.act(feat)

        if self.training:
            mu_curr = feat.mean(dim=(0, 1, 2), keepdim=True)
            self.mu = self.m * self.mu + (1 - self.m) * mu_curr.detach()
        
        feat = feat + self.mu

        feat = feat.permute(0, 3, 1, 2) 
        feat = self.fir(feat)
        feat = feat.permute(0, 2, 3, 1) 

        feat = self.up(feat) * self.scale

        return feat.permute(0, 3, 1, 2) if is_channel_first else feat

class MonaWrappedVSSBlock(nn.Module):
    def __init__(self, original_block, dim):
        super().__init__()
        self.original_block = original_block
        self.mona = MonaAdapter(dim)
        
    def forward(self, x):
        return self.original_block(x) + self.mona(x)

class FM_VMamba(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.farm = FARM(in_channels=3)
        self.vssm = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=num_classes, 
            depths=[2, 2, 9, 2],       
            dims=96,                   
            drop_rate=0.2,             
            drop_path_rate=0.1,        
        )
        
        for i, layer in enumerate(self.vssm.layers):
            dim = self.vssm.dims[i]
            for j, block in enumerate(layer.blocks):
                layer.blocks[j] = MonaWrappedVSSBlock(block, dim=dim)

    def forward(self, x):
        x_farm = self.farm(x)
        return self.vssm(x_farm)


CONFIG = {
    'data_dir': '/root/FM-Vmamba/Vmamba/OriginalData', 
    'batch_size': 64,                 
    'lr': 5e-5,                      
    'num_epochs': 150,
    'num_classes': 6,                
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'val_split': 0.2,
    'save_model_path': 'best_fm_vmamba_laryngeal.pth'
}


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),                  
    transforms.RandomRotation(degrees=15),                 
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    hidden_dir = os.path.join(CONFIG['data_dir'], '.ipynb_checkpoints')
    if os.path.exists(hidden_dir):
        shutil.rmtree(hidden_dir)
        
    def is_valid_file(path):
        if '.ipynb_checkpoints' in path:
            return False
        return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        
    dataset = datasets.ImageFolder(CONFIG['data_dir'], transform=train_transform, is_valid_file=is_valid_file)
    print(f"检测到类别字典: {dataset.class_to_idx}")
    
    class_counts_dict = Counter(dataset.targets)
    class_counts = [class_counts_dict[i] for i in range(CONFIG['num_classes'])]
    print(f"实际读取到的各类别图片总数: {class_counts}")
    
    num_val = int(len(dataset) * CONFIG['val_split'])
    num_train = len(dataset) - num_val
    train_set, val_set = random_split(dataset, [num_train, num_val])
    

    val_set.dataset = copy.copy(dataset)
    val_set.dataset.transform = val_transform


    train_labels = [dataset.targets[i] for i in train_set.indices]
    train_class_counts = Counter(train_labels)
    print(f"训练集真实分布: {train_class_counts}")
    
    class_sample_weights = {cls: 1.0 / count for cls, count in train_class_counts.items()}
    sample_weights = [class_sample_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights), 
        replacement=True
    )

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], sampler=sampler, 
                              num_workers=8, pin_memory=True, persistent_workers=True)
    
    val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=8, pin_memory=True, persistent_workers=True)

    model = FM_VMamba(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    
    for name, param in model.named_parameters():
        param.requires_grad = False
        
        if any(key in name for key in ["classifier", "farm", "mona"]):
            param.requires_grad = True
            
        if "norm" in name.lower() or "bn" in name.lower():
            param.requires_grad = True
            
        if "layers.3" in name: 
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print(f"总参数量: {total_params/1e6:.2f} M")
    print(f"可训练参数: {trainable_params/1e6:.2f} M (占 {trainable_params/total_params*100:.2f}%)")
    print("=" * 60)
    
    print(f"已启用动态平衡采样器，使用标准 CrossEntropyLoss。")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'], weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6)
    
    # scaler = torch.amp.GradScaler('cuda')
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=CONFIG['lr'], 
        epochs=CONFIG['num_epochs'], 
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=100
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0.0
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(CONFIG['device'], non_blocking=True), labels.to(CONFIG['device'], non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            current_lr = optimizer.param_groups[0]['lr']
            train_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.1e}")
            
        train_acc = 100 * correct / total       

        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Val]"):
                images, labels = images.to(CONFIG['device'], non_blocking=True), labels.to(CONFIG['device'], non_blocking=True)
                outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CONFIG['save_model_path'])
            print(f"发现新优模型！已保存最佳模型权重 (Acc: {best_acc:.2f}%)")
            
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()