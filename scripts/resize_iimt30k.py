import os
import sys
from PIL import Image

def process_images(source_dir, target_dir):
    """
    将 source_dir 中的图片缩放到 512x512，
    然后从中心裁剪出高度 48、宽度 512 的区域，
    保存到 target_dir。
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 支持的图片扩展名
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(extensions):
            src_path = os.path.join(source_dir, filename)
            try:
                with Image.open(src_path) as img:
                    # 1. 缩放到 512x512（拉伸，不保持原比例）
                    img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)

                    # 2. 中心裁剪 48x512（高48，宽512）
                    left = 0
                    top = (512 - 48) // 2          # 垂直方向中心点起始
                    right = 512
                    bottom = top + 48
                    img_cropped = img_resized.crop((left, top, right, bottom))

                    # 3. 保存
                    target_path = os.path.join(target_dir, filename)
                    img_cropped.save(target_path)
                    print(f"Processed: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <source_dir> <target_dir>")
        sys.exit(1)

    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    process_images(source_dir, target_dir)