import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import os

def load_image(image_path, scale_percent=25):
    image = cv2.imread(image_path)   #Load image (BGR)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #convert to rgb
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized

def reshape_image(image):
    return image.reshape((-1, 3))

# def compress_with_kmeans(pixels, n_clusters=16):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
#     kmeans.fit(pixels)
#     compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
#     return np.clip(compressed_pixels.astype('uint8'), 0, 255)

def compress_with_kmeans(pixels, n_clusters=16):
    print("Compressing image using KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
    labels = np.asarray(kmeans.labels_, dtype=np.uint8)

    compressed_pixels = centers[labels]  # Replace each pixel with its cluster center
    return compressed_pixels

def compress_with_dbscan(pixels, eps=10, min_samples=50):
    sample_size = min(5000, pixels.shape[0])
    sample_indices = np.random.choice(pixels.shape[0], sample_size, replace=False)
    sampled_pixels = pixels[sample_indices]

    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(sampled_pixels)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(pixels_scaled)

    unique_labels = np.unique(labels[labels != -1])
    print(f"DBSCAN clusters (excluding noise): {len(unique_labels)}")

    new_pixels = np.copy(pixels)

    for label in unique_labels:
        mask = (labels == label)
        cluster_mean = np.mean(sampled_pixels[mask], axis=0).astype('uint8')
        new_pixels[sample_indices[mask]] = cluster_mean

    noise_mask = (labels == -1)
    new_pixels[sample_indices[noise_mask]] = sampled_pixels[noise_mask]

    return np.clip(new_pixels.astype('uint8'), 0, 255)

def reconstruct_image(pixels, original_shape):
    return pixels.reshape(original_shape)

def save_image(image, filename):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_bgr)
    print(f"Saved: {filename}")

def show_images(original, kmeans_compressed, dbscan_compressed):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(original)
    ax[0].set_title("Original Image")
    ax[1].imshow(kmeans_compressed)
    ax[1].set_title("KMeans Compressed")
    ax[2].imshow(dbscan_compressed)
    ax[2].set_title("DBSCAN Compressed")
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = "image.png"  
    output_dir = "compressed_images"
    os.makedirs(output_dir, exist_ok=True)

    image = load_image(image_path, scale_percent=25)
    original_shape = image.shape
    pixels = reshape_image(image)

    kmeans_pixels = compress_with_kmeans(pixels, n_clusters=16)
    kmeans_image = reconstruct_image(kmeans_pixels, original_shape)
    save_image(kmeans_image, os.path.join(output_dir, "kmeans_compressed.png"))

 
    dbscan_pixels = compress_with_dbscan(pixels, eps=10, min_samples=50)
    dbscan_image = reconstruct_image(dbscan_pixels, original_shape)
    save_image(dbscan_image, os.path.join(output_dir, "dbscan_compressed.png"))

    
    show_images(image, kmeans_image, dbscan_image)
