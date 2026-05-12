import os
import json
from collections import Counter
from pathlib import Path

def count_images_by_category(image_dir, output_format='text'):
    """
    统计指定目录下各个类别的图像数量
    
    参数:
        image_dir: 图片目录路径
        output_format: 输出格式，可选 'text', 'json'
    
    返回:
        统计结果
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.webp'}
    
    # 检查目录是否存在
    if not os.path.exists(image_dir):
        return f"错误: 目录 '{image_dir}' 不存在"
    
    if not os.path.isdir(image_dir):
        return f"错误: '{image_dir}' 不是目录"
    
    # 获取所有图片文件
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                # 使用目录名作为类别，如果图片直接在根目录，类别为'root'
                category = os.path.basename(root) if root != image_dir else 'root'
                image_files.append(category)
    
    # 统计每个类别的数量
    counter = Counter(image_files)
    
    # 计算总数
    total_count = sum(counter.values())
    
    # 按数量排序
    sorted_counts = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    
    # 根据输出格式返回结果
    if output_format == 'json':
        result = {
            'total': total_count,
            'categories': sorted_counts,
            'directory': image_dir
        }
        return json.dumps(result, indent=2, ensure_ascii=False)
    else:
        # 文本格式输出
        output = []
        output.append(f"图片目录: {image_dir}")
        output.append("=" * 50)
        
        for category, count in sorted_counts.items():
            percentage = (count / total_count * 100) if total_count > 0 else 0
            output.append(f"{category}: {count}张 ({percentage:.1f}%)")
        
        output.append("-" * 50)
        output.append(f"总计: {total_count}张图片")
        output.append(f"类别数: {len(sorted_counts)}个")
        
        return "\n".join(output)

def count_images_from_paths(file_paths):
    """
    从文件路径列表统计类别
    适用于图片分散在不同目录的情况
    
    参数:
        file_paths: 图片文件路径列表
    
    返回:
        统计结果
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    image_categories = []
    
    for file_path in file_paths:
        if Path(file_path).suffix.lower() in image_extensions:
            # 获取父目录名作为类别
            category = Path(file_path).parent.name
            if category == '':
                category = 'root'
            image_categories.append(category)
    
    # 统计
    counter = Counter(image_categories)
    total_count = sum(counter.values())
    
    # 格式化输出
    output = []
    output.append("图片统计结果:")
    output.append("=" * 40)
    
    for category, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_count * 100) if total_count > 0 else 0
        output.append(f"{category}: {count}张 ({percentage:.1f}%)")
    
    output.append("-" * 40)
    output.append(f"总计: {total_count}张图片")
    output.append(f"类别数: {len(counter)}个")
    
    return "\n".join(output)

# 使用示例
if __name__ == "__main__":
    # 示例1: 统计指定目录下的图片
    # 请将以下路径替换为你的实际图片目录路径
    image_directory = "/root/FM-Vmamba/Vmamba/OriginalData"  # 这里修改为你的图片目录
    
    # 检查目录是否存在
    if os.path.exists(image_directory) and os.path.isdir(image_directory):
        result = count_images_by_category(image_directory, output_format='text')
        print(result)
    else:
        print(f"目录 '{image_directory}' 不存在，请修改路径")
    
    print("\n" + "="*60 + "\n")
    
    # 示例2: 从文件路径列表统计
    # 这里是一些示例路径，你可以替换为实际路径
    example_paths = [
        "./dataset/cat/image1.jpg",
        "./dataset/cat/image2.jpg", 
        "./dataset/dog/image1.png",
        "./dataset/dog/image2.jpg",
        "./dataset/bird/image1.jpeg",
        "./other/image1.png"
    ]
    
    # 注意：这里只是示例，你需要确保这些路径真实存在
    result2 = count_images_from_paths(example_paths)
    print("示例路径统计结果:")
    print(result2)
    
    # 示例3: 输出JSON格式
    if os.path.exists(image_directory) and os.path.isdir(image_directory):
        json_result = count_images_by_category(image_directory, output_format='json')
        print("\nJSON格式输出示例:")
        print(json_result)