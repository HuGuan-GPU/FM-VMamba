import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.fft

# 清理启动前的显存缓存
torch.cuda.empty_cache()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 1. 导入官方 VMamba 模型
# ==========================================
try:
    from vmamba import VSSM 
    print("成功导入官方 VMamba 模型！")
except ImportError as e:
    print(f"导入失败，请检查目录结构或编译环境。错误信息: {e}")
    sys.exit(1)

# ==========================================
# 2. 完整的网络架构
# ==========================================
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

    def forward(self, x, return_vis=False):
        fft_x = torch.fft.fft2(x, norm="ortho")
        fft_x_shifted = torch.fft.fftshift(fft_x) 
        
        amplitude = torch.log1p(torch.abs(fft_x_shifted))

        M_freq = self.freq_controller(amplitude)
        fft_x_filtered = fft_x_shifted * M_freq

        fft_x_ishifted = torch.fft.ifftshift(fft_x_filtered)
        x_restored = torch.fft.ifft2(fft_x_ishifted, norm="ortho").real

        output = x + self.alpha * x_restored 
        output = torch.clamp(output, 0, 1)

        if return_vis:
            return output, amplitude, M_freq
        return output

# class MonaAdapter(nn.Module):
#     def __init__(self, dim, m=0.9):
#         super().__init__()
#         self.dim = dim
#         self.m = m 
#         r = max(16, dim // 8)  
#         self.down = nn.Linear(dim, r, bias=False)
#         self.act = nn.GELU() 
#         self.fir = nn.Conv2d(r, r, kernel_size=3, padding=1, groups=r, bias=False)
#         self.up = nn.Linear(r, dim, bias=False)
#         nn.init.zeros_(self.up.weight)
#         self.register_buffer('mu', torch.zeros(1, 1, 1, r))
#         self.scale = nn.Parameter(torch.ones(1) * 0.1)
class MonaAdapter(nn.Module):
    def __init__(self, dim, m=0.9):
        super().__init__()
        self.dim = dim
        self.m = m 

        r = max(8, dim // 4)  
        
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
    def __init__(self, num_classes=6):
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

# ==========================================
# 3. 辅助函数 (预处理与可视化)
# ==========================================
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size), 
        transforms.ToTensor(),           
    ])
    return transform(image)

# def visualize_comparison(original_img, processed_img, amplitude, mask, class_name, filename, output_dir):
#     fig, axes = plt.subplots(2, 3, figsize=(16, 10))
#     fig.suptitle(f'FARM Module Processing Analysis - {class_name}\n({filename})', fontsize=18, fontweight='bold')
    
#     ori_np = original_img.squeeze().permute(1, 2, 0).cpu().detach().numpy()
#     pro_np = processed_img.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    
#     amp_np = amplitude.squeeze().mean(dim=0).cpu().detach().numpy()
#     mask_np = mask.squeeze().mean(dim=0).cpu().detach().numpy()

#     axes[0, 0].imshow(np.clip(ori_np, 0, 1))
#     axes[0, 0].set_title('1. Original Image (Space Domain)', fontsize=14)
#     axes[0, 0].axis('off')
    
#     axes[0, 1].imshow(np.clip(pro_np, 0, 1))
#     axes[0, 1].set_title('2. Processed by FARM', fontsize=14)
#     axes[0, 1].axis('off')
    
#     diff = np.abs(pro_np - ori_np)
#     im_diff = axes[0, 2].imshow(diff.mean(axis=2), cmap='magma')
#     axes[0, 2].set_title('3. Difference Heatmap\n(Highlights/Noise Suppressed)', fontsize=14)
#     axes[0, 2].axis('off')
#     plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
#     im_amp = axes[1, 0].imshow(amp_np, cmap='viridis')
#     axes[1, 0].set_title('4. Original Spectrum Amplitude\n(Log Scaled)', fontsize=14)
#     axes[1, 0].axis('off')
#     plt.colorbar(im_amp, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
#     im_mask = axes[1, 1].imshow(mask_np, cmap='inferno')
#     axes[1, 1].set_title('5. Learned Frequency Mask ($M_{freq}$)\n(Attention Map)', fontsize=14)
#     axes[1, 1].axis('off')
#     plt.colorbar(im_mask, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
#     from mpl_toolkits.mplot3d import Axes3D
#     H, W = mask_np.shape
#     X, Y = np.meshgrid(np.arange(W), np.arange(H))
#     ax3d = fig.add_subplot(2, 3, 6, projection='3d')
#     stride = 4
#     surf = ax3d.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], mask_np[::stride, ::stride], 
#                              cmap='inferno', alpha=0.9, linewidth=0, antialiased=True)
#     ax3d.set_title('6. 3D View of Frequency Mask', fontsize=14)
#     ax3d.view_init(elev=30, azim=45)
    
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.88)
    
#     save_path = os.path.join(output_dir, f"{filename}_academic_analysis.png")
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     plt.close(fig) 
#     return save_path
def visualize_comparison(original_img, processed_img, amplitude, mask, class_name, filename, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'FARM Module Processing Analysis - {class_name}\n({filename})', fontsize=18, fontweight='bold')
    
    ori_np = original_img.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    pro_np = processed_img.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    
    amp_np = amplitude.squeeze().mean(dim=0).cpu().detach().numpy()
    mask_np = mask.squeeze().mean(dim=0).cpu().detach().numpy()

    # 1. 空间域原图
    axes[0, 0].imshow(np.clip(ori_np, 0, 1))
    axes[0, 0].set_title('1. Original Image (Space Domain)', fontsize=14)
    axes[0, 0].axis('off')
    
    # 2. FARM 处理后
    axes[0, 1].imshow(np.clip(pro_np, 0, 1))
    axes[0, 1].set_title('2. Processed by FARM', fontsize=14)
    axes[0, 1].axis('off')
    
    # 3. 差异热力图 (动态自适应拉伸)
    diff = np.abs(pro_np - ori_np)
    diff_mean = diff.mean(axis=2)
    # 增加微小差异的可视化对比度
    im_diff = axes[0, 2].imshow(diff_mean, cmap='magma', vmin=0, vmax=np.max(diff_mean))
    axes[0, 2].set_title('3. Residual Enhancement Map\n(Micro-textures Highlighted)', fontsize=14)
    axes[0, 2].axis('off')
    plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 4. 原始频谱图
    vmax_amp = np.percentile(amp_np, 99.5) 
    im_amp = axes[1, 0].imshow(amp_np, cmap='viridis', vmin=0, vmax=vmax_amp)
    axes[1, 0].set_title('4. Original Spectrum Amplitude\n(Log Scaled, Center Suppressed)', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im_amp, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 5. 注意力掩膜 (动态自适应映射)
    im_mask = axes[1, 1].imshow(mask_np, cmap='inferno')
    axes[1, 1].set_title('5. Learned Frequency Mask ($M_{freq}$)\n(Attention Map)', fontsize=14)
    axes[1, 1].axis('off')
    plt.colorbar(im_mask, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 6. 3D 掩膜视图
    from mpl_toolkits.mplot3d import Axes3D
    H, W = mask_np.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    ax3d = fig.add_subplot(2, 3, 6, projection='3d')
    stride = 4
    surf = ax3d.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], mask_np[::stride, ::stride], 
                             cmap='inferno', alpha=0.9, linewidth=0, antialiased=True)
    ax3d.set_title('6. 3D View of Learned Features', fontsize=14)
    # 调整视角，让双峰看得更清楚
    ax3d.view_init(elev=25, azim=60) 
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    save_path = os.path.join(output_dir, f"{filename}_academic_analysis.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig) 
    return save_path
# ==========================================
# 4. 测试主流程
# ==========================================
def test_farm_module_on_dataset(input_dir="test_image", output_dir="FARM_result_image"):
    print("=" * 60)
    print("FARM(傅里叶自适应恢复模块)可视化验证")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("正在加载已训练的完整 FM_VMamba 模型权重...")
    
    # 1. 实例化完整的网络
    full_model = FM_VMamba(num_classes=6).to(device)
    
    # 2. 读取训练保存的最佳权重
    weight_path = '/root/FM-Vmamba/Vmamba/VMamba-main/best_vmamba_laryngeal.pth'
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device)
        full_model.load_state_dict(state_dict)
        print(f"成功加载权重文件：{weight_path}")
    else:
        print(f"错误：找不到 {weight_path}！请确保权重文件在当前目录下。")
        return
        
    full_model.eval()
    
    # 3. 提取出已经加载了成熟权重的 FARM 模块用于测试
    model = full_model.farm
    # 查看网络学到的残差系数
    print(f"当前模型 FARM 模块学到的 Alpha (残差融合系数) 为: {model.alpha.item():.6f}")
    processed_count = 0
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                img_path = os.path.join(input_dir, filename)
                base_name = os.path.splitext(filename)[0]
                
                lesion_class = base_name.split('_')[0] if '_' in base_name else "Sample"
                
                print(f"正在深度分析: {filename}...")
                input_tensor = load_and_preprocess_image(img_path)
                input_batch = input_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output_tensor, amplitude, freq_mask = model(input_batch, return_vis=True)
                
                save_path = visualize_comparison(
                    input_batch, output_tensor, amplitude, freq_mask, 
                    lesion_class, base_name, output_dir
                )
                
                processed_count += 1
                print(f"可视化已保存: {save_path}")
                
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

    print("\n" + "=" * 60)
    print(f"验证完成！共处理 {processed_count} 张图片。")
    print(f"所有配图均已保存在 '{output_dir}' 文件夹下！")
    print("=" * 60)

if __name__ == "__main__":
    # 配置中文字体防乱码
    font_path = "/root/FM-Vmamba/SIMHEI.TTF"  
    if os.path.exists(font_path):
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        
    test_dir = "/root/FM-Vmamba/Vmamba/VMamba-main/test_image"
    if not os.path.exists(test_dir):
        print(f"找不到测试图文件夹 '{test_dir}'。")
    else:
        test_farm_module_on_dataset(input_dir=test_dir, output_dir="FARM_result_image")