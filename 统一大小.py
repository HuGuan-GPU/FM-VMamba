import os
import cv2
from pathlib import Path

def resize_images(input_dir, output_dir, target_size=(256, 256)):
    """
    将输入目录中的所有图片调整为指定大小，并保存到输出目录
    
    Args:
        input_dir: 输入图片目录路径
        output_dir: 输出图片目录路径
        target_size: 目标图片大小，默认为(256, 256)
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    
    # 遍历输入目录
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # 读取图片
                img = cv2.imread(input_path)
                if img is None:
                    print(f"警告: 无法读取图片 {filename}")
                    continue
                
                # 调整图片大小
                resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                
                # 保存图片
                cv2.imwrite(output_path, resized_img)
                print(f"已处理: {filename}")
                
            except Exception as e:
                print(f"处理图片 {filename} 时出错: {e}")

if __name__ == "__main__":
    # 输入和输出目录
    input_directory = "test_image"
    output_directory = "test_image_256"
    
    # 检查输入目录是否存在
    if not os.path.exists(input_directory):
        print(f"错误: 目录 '{input_directory}' 不存在")
    else:
        # 执行图片大小调整
        resize_images(input_directory, output_directory)
        print(f"所有图片已处理完成，保存到 '{output_directory}' 目录")