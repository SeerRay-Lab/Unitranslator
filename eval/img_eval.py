import os
from tqdm import tqdm
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
import multiprocessing as mp
import argparse

def split_image_into_patches(img, patch_size):
    """Splits an image into smaller patches."""
    patches = []
    height, width, _ = img.shape
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            # Ensure the patch is the correct size, otherwise, ssim can fail
            ph, pw, _ = patch.shape
            if ph != patch_size or pw != patch_size:
                # Pad the patch if it's smaller than the desired size
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded_patch[0:ph, 0:pw] = patch
                patch = padded_patch
            patches.append(patch)
    return patches

def calculate_ssim_for_files(g_file, r_file, patch_size):
    """Reads, resizes, and calculates the mean SSIM for patches of two images."""
    try:
        g_img = cv2.resize(cv2.imread(g_file), (512, 512))
        r_img = cv2.resize(cv2.imread(r_file), (512, 512))
        
        g_patches = split_image_into_patches(g_img, patch_size)
        r_patches = split_image_into_patches(r_img, patch_size)
        
        patch_ssims = []
        for g_patch, r_patch in zip(g_patches, r_patches):
            # The 'multichannel' argument is deprecated; 'channel_axis' is the correct replacement.
            patch_ssim = compare_ssim(g_patch, r_patch, channel_axis=-1)
            patch_ssims.append(patch_ssim)
            
        return np.mean(patch_ssims)
    except Exception as e:
        print(f"Error processing files {g_file} and {r_file}: {e}")
        return 0.0 # Return a neutral value on error

def process_files(file_pair):
    """Wrapper function to process a single pair of files for multiprocessing."""
    g_file, r_file = file_pair
    # The global patch_sizes variable will be accessible by child processes
    ssims = {
        patch_size: calculate_ssim_for_files(g_file, r_file, patch_size) 
        for patch_size in patch_sizes
    }
    return ssims

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate SSIM between two image directories by matching filenames.")
    parser.add_argument("--gen_dir", type=str, default="results/outputs_3b_tf_mask_en", help="Directory with generated images.")
    parser.add_argument("--ref_dir", type=str, default="/mnt/vlm-ks3/ljh/data/translationV/iwslt14.de-en-images/test_en", help="Directory with reference images.")
    parser.add_argument("--workers", type=int, default=64, help="Number of worker processes for multiprocessing.")
    args = parser.parse_args()

    # This will be globally available to worker processes spawned by mp.Pool
    patch_sizes = [512]

    # --- MODIFICATION START ---

    # 1. Create a map of generated files for quick lookup
    print(f"Scanning generated images in: {args.gen_dir}")
    generate_files_map = {
        f: os.path.join(args.gen_dir, f)
        for f in os.listdir(args.gen_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    }

    # 2. Find matching file pairs based on filename
    print(f"Scanning reference images in: {args.ref_dir}")
    ref_files = [f for f in os.listdir(args.ref_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    file_pairs = []
    for ref_filename in ref_files:
        if ref_filename in generate_files_map:
            gen_filepath = generate_files_map[ref_filename]
            ref_filepath = os.path.join(args.ref_dir, ref_filename)
            file_pairs.append((gen_filepath, ref_filepath))
        else:
            print(f"Warning: No match found for reference file: {ref_filename}")

    # --- MODIFICATION END ---
    
    if not file_pairs:
        print("No matching image pairs were found. Exiting.")
        exit()

    ssims = {patch_size: [] for patch_size in patch_sizes}
    
    print(f"\nFound {len(file_pairs)} matching image pairs to process.")
    print(f"Using {args.workers} worker processes (CPU count: {mp.cpu_count()})...")

    with mp.Pool(args.workers) as pool:
        # Use the matched file_pairs list and update tqdm's total
        for file_ssims in tqdm(pool.imap(process_files, file_pairs), total=len(file_pairs)):
            for patch_size in patch_sizes:
                ssims[patch_size].append(file_ssims[patch_size])

    print("-" * 30)
    average_ssims = {patch_size: np.mean(ssims[patch_size]) for patch_size in patch_sizes}
    for patch_size in patch_sizes:
        print(f"Average SSIM for patch size {patch_size}: {average_ssims[patch_size]:.6f}")

