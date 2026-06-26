import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm # 用于显示一个漂亮的进度条

def generate_and_save_mask(input_path, output_path, threshold_val, use_smooth, kernel_size):
    """
    从单个图像文件生成Mask，并将其保存到指定路径。
    """
    # 1. 以灰度模式读取图像
    gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        # 如果文件无法读取，打印警告并跳过
        print(f"\n警告：无法读取文件或文件非图像格式，已跳过: {os.path.basename(input_path)}")
        return False

    # 2. 应用反向二值阈值处理
    # 这一步将深色区域（文字）变为白色（255），浅色区域（背景）变为黑色（0）
    _, binary_mask = cv2.threshold(gray_image, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # 3. (可选) 应用形态学操作平滑边缘
    if use_smooth:
        # 确保核大小为奇数，这是OpenCV的要求
        k_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        # 开运算：移除小的白色噪点和边缘的毛刺
        opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # 闭运算：填充文字内部的小黑洞，并使边缘更平滑
        smoothed_mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
        final_mask = smoothed_mask
    else:
        final_mask = binary_mask

    # 4. 保存最终生成的Mask图像
    cv2.imwrite(output_path, final_mask)
    return True

def main():
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="批量为目录中的图像生成Mask。将深色文字转为白色，浅色背景转为黑色。"
    )
    
    parser.add_argument('input_dir', type=str, help="包含输入图像的目录路径。")
    parser.add_argument('output_dir', type=str, help="用于保存生成的Mask的目录路径。")
    
    parser.add_argument('--threshold', type=int, default=127,
                        help="用于区分文字和背景的灰度阈值 (0-255)。默认值: 127。")
    
    parser.add_argument('--smooth', action='store_true',
                        help="添加此标志以应用形态学操作来平滑Mask边缘。")
                        
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="用于平滑操作的核大小 (应为奇数)。默认值: 5。")
                        
    args = parser.parse_args()

    # --- 验证和准备目录 ---
    
    # 1. 检查输入目录是否存在
    if not os.path.isdir(args.input_dir):
        print(f"错误：找不到输入目录 '{args.input_dir}'")
        return

    # 2. 如果输出目录不存在，则创建它
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    if args.smooth:
        print(f"边缘平滑已启用，核大小: {args.kernel_size}")

    # 3. 查找所有支持的图像文件
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(supported_extensions)]

    if not image_files:
        print("在输入目录中未找到支持的图像文件。")
        return
        
    print(f"找到 {len(image_files)} 张图像进行处理。")

    # --- 开始批量处理 ---
    
    # 使用 tqdm 创建一个进度条
    for filename in tqdm(image_files, desc="正在处理图像"):
        # 构建完整的文件路径
        input_path = os.path.join(args.input_dir, filename)
        
        # 构建输出文件的路径，统一保存为.png格式
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_mask.png"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # 调用函数生成并保存Mask
        generate_and_save_mask(
            input_path,
            output_path,
            args.threshold,
            args.smooth,
            args.kernel_size
        )

    print("\n所有图像处理完成！")

if __name__ == "__main__":
    main()

