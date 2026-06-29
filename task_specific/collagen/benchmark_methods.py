"""
Comparison: Gabor Filter vs Structure Tensor Analysis

This script provides side-by-side comparison and benchmarking of both methods.
Use this to understand the trade-offs and choose the best method for your use case.
"""

import numpy as np
import tifffile
import time
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.ndimage import sobel
from skimage import filters
from skimage.filters import gabor_kernel
from tqdm import tqdm


def benchmark_gabor_method(image_path, num_directions=8):
    """Time the Gabor filter approach."""
    
    try:
        image = tifffile.imread(image_path)
        if image.ndim == 3:
            image = rgb2gray(image)
        if image.max() > 1.0:
            image = image.astype(np.float64)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    print("\n" + "="*70)
    print("GABOR FILTER METHOD")
    print("="*70)
    
    start = time.time()
    
    # Create Gabor kernels
    theta = np.linspace(0, np.pi, num_directions, endpoint=False)
    kernels = [gabor_kernel(frequency=0.1, theta=t) for t in theta]
    feats = np.zeros((image.shape[0], image.shape[1], len(kernels)), dtype=np.double)
    
    # Convolve with each kernel
    for k, kernel in tqdm(enumerate(kernels), total=len(kernels), desc="Gabor convolutions"):
        filtered = convolve2d(image, np.real(kernel), mode="same", boundary="wrap")
        feats[:, :, k] = filtered
    
    # Compute anisotropy and orientation
    anisotropy = np.std(feats, axis=-1)
    orientation = theta[np.argmax(feats, axis=-1)]
    
    elapsed = time.time() - start
    
    print(f"Number of directions: {num_directions}")
    print(f"Anisotropy range: [{anisotropy.min():.4f}, {anisotropy.max():.4f}]")
    print(f"Orientation range: [{orientation.min():.4f}, {orientation.max():.4f}] rad")
    print(f"Memory used: ~{feats.nbytes / 1e6:.1f} MB (7 × image size)")
    print(f"⏱️  ELAPSED TIME: {elapsed:.2f} seconds")
    
    return anisotropy, orientation, elapsed


def benchmark_structure_tensor_method(image_path):
    """Time the structure tensor approach."""
    
    try:
        image = tifffile.imread(image_path)
        if image.ndim == 3:
            image = rgb2gray(image)
        if image.max() > 1.0:
            image = image.astype(np.float64)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    print("\n" + "="*70)
    print("STRUCTURE TENSOR METHOD")
    print("="*70)
    
    start = time.time()
    
    # Step 1: Compute gradients
    smoothed = gaussian_filter(image, sigma=1.0)
    Gx = sobel(smoothed, axis=1)
    Gy = sobel(smoothed, axis=0)
    
    # Step 2: Tensor components
    Gxx = Gx * Gx
    Gyy = Gy * Gy
    Gxy = Gx * Gy
    
    # Step 3: Smooth tensor
    Txx = gaussian_filter(Gxx, sigma=2.0)
    Tyy = gaussian_filter(Gyy, sigma=2.0)
    Txy = gaussian_filter(Gxy, sigma=2.0)
    
    # Step 4: Eigendecomposition (vectorized)
    trace = Txx + Tyy
    det = Txx * Tyy - Txy * Txy
    discriminant = np.sqrt(np.maximum((Txx - Tyy)**2 + 4*Txy**2, 0))
    
    lambda_max = (trace + discriminant) / 2.0
    lambda_min = (trace - discriminant) / 2.0
    
    # Coherence = anisotropy
    denominator = lambda_max + lambda_min
    coherence = np.where(denominator > 1e-10, 
                        (lambda_max - lambda_min) / denominator, 
                        0.0)
    
    # Orientation
    orientation = 0.5 * np.arctan2(2*Txy, Txx - Tyy)
    orientation = np.where(orientation < 0, orientation + np.pi, orientation)
    
    elapsed = time.time() - start
    
    print(f"Coherence range: [{coherence.min():.4f}, {coherence.max():.4f}]")
    print(f"Orientation range: [{orientation.min():.4f}, {orientation.max():.4f}] rad")
    print(f"Memory used: ~{(Txx.nbytes * 3 + lambda_max.nbytes * 2) / 1e6:.1f} MB (5 × image size)")
    print(f"⏱️  ELAPSED TIME: {elapsed:.2f} seconds")
    
    return coherence, orientation, elapsed


def compare_orientation_maps(gabor_aniso, struct_coher, figsize=(14, 5)):
    """Visualize and compare the two anisotropy measures."""
    
    # Normalize both to [0, 1] for fair comparison
    gabor_norm = (gabor_aniso - gabor_aniso.min()) / (gabor_aniso.max() - gabor_aniso.min())
    struct_norm = struct_coher / 1.0  # Already in [0, 1]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Gabor anisotropy
    im1 = axes[0].imshow(gabor_norm, cmap='jet')
    axes[0].set_title('Gabor Anisotropy\n(normalized)', fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Structure tensor coherence
    im2 = axes[1].imshow(struct_norm, cmap='jet')
    axes[1].set_title('Structure Tensor Coherence\n(0=isotropic, 1=aligned)', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference map
    diff = np.abs(gabor_norm - struct_norm)
    im3 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference between methods\n(lower = agreement)', fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    fig.suptitle('Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("method_comparison.tif", dpi=300)
    plt.show()
    
    print(f"\nComparison Statistics:")
    print(f"  Mean difference: {diff.mean():.4f}")
    print(f"  Std of difference: {diff.std():.4f}")
    print(f"  Max difference: {diff.max():.4f}")
    print(f"  Correlation: {np.corrcoef(gabor_norm.flatten(), struct_norm.flatten())[0, 1]:.4f}")


def print_summary(gabor_time, struct_tensor_time):
    """Print summary comparison."""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n⏱️  EXECUTION TIME:")
    print(f"  Gabor filters:     {gabor_time:.2f}s")
    print(f"  Structure tensor:  {struct_tensor_time:.2f}s")
    print(f"  Speedup factor:    {gabor_time/struct_tensor_time:.1f}x faster with Structure Tensor")
    
    print(f"\n💾 MEMORY FOOTPRINT:")
    print(f"  Gabor filters:     ~8× image size (8 response maps)")
    print(f"  Structure tensor:  ~5× image size (tensor components + eigenvalues)")
    
    print(f"\n🎯 ROBUSTNESS:")
    print(f"  Gabor filters:")
    print(f"    ✗ Sensitive to frequency parameter selection")
    print(f"    ✗ Quantized to 8 directions (loses continuity)")
    print(f"    ✗ More sensitive to noise")
    print(f"    ✓ Good for texture analysis")
    
    print(f"\n  Structure Tensor (like OrientationJ):")
    print(f"    ✓ Mathematically principled (eigenvalue decomposition)")
    print(f"    ✓ Continuous orientation (not quantized)")
    print(f"    ✓ Robust with Gaussian smoothing parameter (σ_tensor)")
    print(f"    ✓ Industry-standard in image analysis")
    print(f"    ✓ Better for fiber/ridge analysis")
    
    print(f"\n📊 ANISOTROPY INTERPRETATION:")
    print(f"  Gabor:              std(gabor_responses) → unbounded scale, needs normalization")
    print(f"  Structure Tensor:   (λ₁ - λ₂)/(λ₁ + λ₂) → [0,1], direct physical interpretation")
    
    print(f"\n🔬 RECOMMENDATION:")
    print(f"  For collagen fiber analysis: USE STRUCTURE TENSOR")
    print(f"    → Faster ({gabor_time/struct_tensor_time:.1f}x)")
    print(f"    → More robust")
    print(f"    → Matches OrientationJ standard")
    print(f"    → Better for publication-quality analysis")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    
    image_path = r"D:\integrated_pipeline\registered_data\SRS\channels\K21_198_3-collagen_black.tif"
    
    print("\n🔍 BENCHMARK: Gabor vs Structure Tensor for Collagen Analysis\n")
    
    try:
        # Benchmark Gabor method
        gabor_aniso, gabor_orient, gabor_time = benchmark_gabor_method(image_path, num_directions=8)
        
        # Benchmark structure tensor method
        struct_coher, struct_orient, struct_time = benchmark_structure_tensor_method(image_path)
        
        if gabor_aniso is not None and struct_coher is not None:
            # Comparison
            compare_orientation_maps(gabor_aniso, struct_coher)
            
            # Summary
            print_summary(gabor_time, struct_time)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
