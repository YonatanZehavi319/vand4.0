import numpy as np
from PIL import Image
import random
import cv2
import argparse
import os


def rand_perlin_2d(shape, res=(4, 4)):
    """Generate 2D Perlin noise."""
    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat(d[0], 0).repeat(d[1], 1)
    dot = lambda grad, shift: (
        np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]), axis=-1)
        * grad[:shape[0], :shape[1]]
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

    t = f(grid[:shape[0], :shape[1]])
    return np.sqrt(2) * (
        (1 - t[..., 0]) * ((1 - t[..., 1]) * n00 + t[..., 1] * n01)
        + t[..., 0] * ((1 - t[..., 1]) * n10 + t[..., 1] * n11)
    )


def generate_perlin_mask(h, w, threshold=0.5, res_range=(2, 6)):
    """Generate a binary mask using Perlin noise."""
    res_y = random.randint(*res_range)
    res_x = random.randint(*res_range)
    h_adj = (h // res_y) * res_y
    w_adj = (w // res_x) * res_x
    noise = rand_perlin_2d((h_adj, w_adj), (res_y, res_x))
    noise = cv2.resize(noise, (w, h))
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    mask = (noise > threshold).astype(np.float32)
    return mask


def synthesize_anomaly_texture(image, texture_source=None, opacity_range=(0.5, 1.0)):
    """Perlin mask + external texture blend."""
    h, w = image.shape[:2]
    mask = generate_perlin_mask(h, w)

    if texture_source is None:
        texture = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    else:
        texture = cv2.resize(texture_source, (w, h))

    opacity = random.uniform(*opacity_range)
    mask_3c = np.stack([mask] * 3, axis=-1)

    anomalous = image.copy().astype(np.float32)
    anomalous = anomalous * (1 - mask_3c * opacity) + texture.astype(np.float32) * mask_3c * opacity
    anomalous = np.clip(anomalous, 0, 255).astype(np.uint8)

    return anomalous, mask


def get_object_mask(image):
    """Detect object region using spatial prior + K-means.
    Assumes image edges are background and center is foreground,
    then uses K-means clustering to propagate to the full image."""
    from sklearn.cluster import KMeans

    h, w = image.shape[:2]
    # Work on a smaller version for speed
    small = cv2.resize(image, (64, 64))
    pixels = small.reshape(-1, 3).astype(np.float32)

    # K-means into 2 clusters
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels).reshape(64, 64)

    # Spatial prior: border = background, center = foreground
    border_ratio = 0.05
    bh, bw = max(1, int(64 * border_ratio)), max(1, int(64 * border_ratio))
    border_mask = np.zeros((64, 64), dtype=bool)
    border_mask[:bh, :] = True
    border_mask[-bh:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True

    # Which cluster dominates the border? That's background
    border_labels = labels[border_mask]
    bg_cluster = np.bincount(border_labels).argmax()

    # Object = not background cluster
    obj_small = ((labels != bg_cluster) * 255).astype(np.uint8)

    # Upscale back to original resolution
    obj_mask = cv2.resize(obj_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel)
    return obj_mask  # 255 = object, 0 = background


def _patch_on_object(y, x, ph, pw, obj_mask, min_coverage=0.4):
    """Check if a patch at (y, x) is mostly on the object."""
    region = obj_mask[y:y + ph, x:x + pw]
    return region.mean() / 255.0 >= min_coverage


def synthesize_anomaly_background(image, patch_size_range=(0.02, 0.08), num_patches=1):
    """Copy a patch from one region and paste it at a different location.
    Only picks and pastes patches on the object (not background)."""
    h, w = image.shape[:2]
    anomalous = image.copy()
    mask = np.zeros((h, w), dtype=np.float32)
    obj_mask = get_object_mask(image)

    for _ in range(num_patches):
        frac = random.uniform(*patch_size_range)
        ph = int(h * frac)
        pw = int(w * frac)

        # Source location — must be on the object
        found_source = False
        for _ in range(50):
            sy = random.randint(0, h - ph)
            sx = random.randint(0, w - pw)
            if _patch_on_object(sy, sx, ph, pw, obj_mask):
                found_source = True
                break
        if not found_source:
            continue

        # Target location — must be on the object and away from source
        found_target = False
        for _ in range(50):
            ty = random.randint(0, h - ph)
            tx = random.randint(0, w - pw)
            if not _patch_on_object(ty, tx, ph, pw, obj_mask):
                continue
            overlap_y = max(0, min(sy + ph, ty + ph) - max(sy, ty))
            overlap_x = max(0, min(sx + pw, tx + pw) - max(sx, tx))
            overlap = overlap_y * overlap_x
            if overlap < 0.3 * ph * pw:
                found_target = True
                break
        if not found_target:
            continue

        patch = image[sy:sy + ph, sx:sx + pw].copy()

        # Optional slight transform
        transform = random.choice(['none', 'flip_h', 'flip_v', 'rotate90'])
        if transform == 'flip_h':
            patch = patch[:, ::-1]
        elif transform == 'flip_v':
            patch = patch[::-1, :]
        elif transform == 'rotate90':
            patch = np.rot90(patch)
            patch = cv2.resize(patch, (pw, ph))

        # Soft-edge mask for blending
        patch_mask = np.ones((ph, pw), dtype=np.float32)
        border = max(1, min(ph, pw) // 6)
        patch_mask[:border, :] *= np.linspace(0, 1, border)[:, None]
        patch_mask[-border:, :] *= np.linspace(1, 0, border)[:, None]
        patch_mask[:, :border] *= np.linspace(0, 1, border)[None, :]
        patch_mask[:, -border:] *= np.linspace(1, 0, border)[None, :]

        # Paste
        patch_mask_3c = np.stack([patch_mask] * 3, axis=-1)
        region = anomalous[ty:ty + ph, tx:tx + pw].astype(np.float32)
        blended = region * (1 - patch_mask_3c) + patch.astype(np.float32) * patch_mask_3c
        anomalous[ty:ty + ph, tx:tx + pw] = np.clip(blended, 0, 255).astype(np.uint8)

        mask[ty:ty + ph, tx:tx + pw] = np.maximum(mask[ty:ty + ph, tx:tx + pw], patch_mask)

    mask = (mask > 0.5).astype(np.float32)
    return anomalous, mask


def synthesize_anomaly_crack(image, thickness_range=(1, 3), num_steps_range=(40, 120),
                             darken_range=(0.3, 0.7)):
    """Generate a crack-like anomaly using a random walk on the object.

    Args:
        image: numpy array (H, W, 3), uint8
        thickness_range: line thickness in pixels
        num_steps_range: how many steps the random walk takes (controls crack length)
        darken_range: how much to darken the crack (0=black, 1=no change)
    """
    h, w = image.shape[:2]
    anomalous = image.copy()
    obj_mask = get_object_mask(image)
    mask = np.zeros((h, w), dtype=np.float32)

    # Find a starting point on the object
    object_points = np.argwhere(obj_mask > 0)  # (y, x) pairs
    if len(object_points) == 0:
        return anomalous, mask

    # Pick random start on the object
    start_idx = random.randint(0, len(object_points) - 1)
    cy, cx = object_points[start_idx]

    # Random walk to create a jagged crack path
    num_steps = random.randint(*num_steps_range)
    thickness = random.randint(*thickness_range)
    darken = random.uniform(*darken_range)

    # Pick a general direction with some randomness
    angle = random.uniform(0, 2 * np.pi)
    points = [(cx, cy)]

    for _ in range(num_steps):
        # Step in the general direction with jitter
        angle += random.gauss(0, 0.5)  # jitter the direction
        step_size = random.uniform(1, 4)
        nx = int(cx + step_size * np.cos(angle))
        ny = int(cy + step_size * np.sin(angle))

        # Stay within image and on the object
        nx = np.clip(nx, 0, w - 1)
        ny = np.clip(ny, 0, h - 1)
        if obj_mask[ny, nx] == 0:
            # Hit background, try to steer back
            angle += random.choice([-0.8, 0.8])
            continue

        cx, cy = nx, ny
        points.append((cx, cy))

    if len(points) < 2:
        return anomalous, mask

    # Draw the crack
    pts = np.array(points, dtype=np.int32)
    # Draw on mask
    cv2.polylines(mask, [pts], isClosed=False, color=1.0, thickness=thickness + 2)
    # Blur the mask slightly for softer edges
    mask = cv2.GaussianBlur(mask, (5, 5), 1.0)

    # Darken the crack region
    mask_3c = np.stack([mask] * 3, axis=-1)
    anomalous = anomalous.astype(np.float32)
    anomalous = anomalous * (1 - mask_3c * (1 - darken))
    anomalous = np.clip(anomalous, 0, 255).astype(np.uint8)

    mask = (mask > 0.1).astype(np.float32)
    return anomalous, mask


def synthesize_anomaly_adaptive(image, patch_frac_range=(0.05, 0.15), num_patches=1):
    """Copy-paste anomaly sized relative to individual detected objects.
    Automatically finds objects via connected components and scales patch size
    to fit each object, regardless of how small it is in the image."""
    h, w = image.shape[:2]
    anomalous = image.copy()
    mask = np.zeros((h, w), dtype=np.float32)
    obj_mask = get_object_mask(image)

    # Find individual objects via connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)

    # Filter out background (label 0) and tiny noise components
    min_area = h * w * 0.001  # ignore components smaller than 0.1% of image
    valid_objects = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_objects.append(i)

    if not valid_objects:
        return anomalous, mask

    for _ in range(num_patches):
        # Pick a random object
        obj_id = random.choice(valid_objects)
        ox = stats[obj_id, cv2.CC_STAT_LEFT]
        oy = stats[obj_id, cv2.CC_STAT_TOP]
        ow = stats[obj_id, cv2.CC_STAT_WIDTH]
        oh = stats[obj_id, cv2.CC_STAT_HEIGHT]
        obj_region = (labels == obj_id).astype(np.uint8) * 255

        # Size patch relative to this object
        frac = random.uniform(*patch_frac_range)
        ph = max(4, int(oh * frac))
        pw = max(4, int(ow * frac))

        # Source location — must be on this object
        found_source = False
        for _ in range(50):
            sy = random.randint(oy, min(oy + oh - ph, h - ph))
            sx = random.randint(ox, min(ox + ow - pw, w - pw))
            region = obj_region[sy:sy + ph, sx:sx + pw]
            if region.mean() / 255.0 >= 0.4:
                found_source = True
                break
        if not found_source:
            continue

        # Target location — must be on this object and away from source
        found_target = False
        for _ in range(50):
            ty = random.randint(oy, min(oy + oh - ph, h - ph))
            tx = random.randint(ox, min(ox + ow - pw, w - pw))
            region = obj_region[ty:ty + ph, tx:tx + pw]
            if region.mean() / 255.0 < 0.4:
                continue
            overlap_y = max(0, min(sy + ph, ty + ph) - max(sy, ty))
            overlap_x = max(0, min(sx + pw, tx + pw) - max(sx, tx))
            overlap = overlap_y * overlap_x
            if overlap < 0.3 * ph * pw:
                found_target = True
                break
        if not found_target:
            continue

        patch = image[sy:sy + ph, sx:sx + pw].copy()

        # Optional slight transform
        transform = random.choice(['none', 'flip_h', 'flip_v', 'rotate90'])
        if transform == 'flip_h':
            patch = patch[:, ::-1]
        elif transform == 'flip_v':
            patch = patch[::-1, :]
        elif transform == 'rotate90':
            patch = np.rot90(patch)
            patch = cv2.resize(patch, (pw, ph))

        # Soft-edge mask for blending
        patch_mask = np.ones((ph, pw), dtype=np.float32)
        border = max(1, min(ph, pw) // 6)
        patch_mask[:border, :] *= np.linspace(0, 1, border)[:, None]
        patch_mask[-border:, :] *= np.linspace(1, 0, border)[:, None]
        patch_mask[:, :border] *= np.linspace(0, 1, border)[None, :]
        patch_mask[:, -border:] *= np.linspace(1, 0, border)[None, :]

        # Clip patch_mask to object shape
        obj_clip = obj_region[ty:ty + ph, tx:tx + pw].astype(np.float32) / 255.0
        patch_mask = patch_mask * obj_clip

        # Paste
        patch_mask_3c = np.stack([patch_mask] * 3, axis=-1)
        region = anomalous[ty:ty + ph, tx:tx + pw].astype(np.float32)
        blended = region * (1 - patch_mask_3c) + patch.astype(np.float32) * patch_mask_3c
        anomalous[ty:ty + ph, tx:tx + pw] = np.clip(blended, 0, 255).astype(np.uint8)

        mask[ty:ty + ph, tx:tx + pw] = np.maximum(mask[ty:ty + ph, tx:tx + pw], patch_mask)

    mask = (mask > 0.5).astype(np.float32)
    return anomalous, mask


def synthesize_anomaly(image, mode='background', **kwargs):
    """Generate a synthetic anomaly.

    Modes:
        'background' - copy-paste from same image
        'texture'    - Perlin noise + random texture
        'crack'      - thin jagged crack line on the object
        'adaptive'   - copy-paste sized relative to individual detected objects
        'mixed'      - randomly pick one
    """
    if mode == 'mixed':
        mode = random.choice(['background', 'texture', 'crack', 'adaptive'])

    if mode == 'background':
        return synthesize_anomaly_background(image, **kwargs)
    elif mode == 'texture':
        return synthesize_anomaly_texture(image, **kwargs)
    elif mode == 'crack':
        return synthesize_anomaly_crack(image, **kwargs)
    elif mode == 'adaptive':
        return synthesize_anomaly_adaptive(image, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--mode', type=str, default='background', choices=['background', 'texture', 'crack', 'adaptive', 'mixed'])
    parser.add_argument('--output_dir', type=str, default='./synth_demo')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image = cv2.imread(args.image)
    anomalous, mask = synthesize_anomaly(image, mode=args.mode)

    basename = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.output_dir, f'{basename}_original.png'), image)
    cv2.imwrite(os.path.join(args.output_dir, f'{basename}_{args.mode}.png'), anomalous)
    cv2.imwrite(os.path.join(args.output_dir, f'{basename}_mask.png'), (mask * 255).astype(np.uint8))

    # Also save the object mask so you can verify Otsu segmentation
    obj_mask = get_object_mask(image)
    cv2.imwrite(os.path.join(args.output_dir, f'{basename}_object_mask.png'), obj_mask)

    print(f"Saved to {args.output_dir}/")
    print(f"  {basename}_original.png")
    print(f"  {basename}_{args.mode}.png")
    print(f"  {basename}_mask.png")
    print(f"  {basename}_object_mask.png  (Otsu object detection)")
