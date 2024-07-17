import numpy as np
import matplotlib.pyplot as plt
import rasterio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans 
from sklearn.mixture import GaussianMixture

# Within Class Sum of Squares
def calculate_wcss(sampled_data, max_clusters):
    """
    Calculate the Within-Cluster Sum of Squares (WCSS) for a range of cluster numbers using the K-means algorithm.

    Args:
        sampled_data (array-like): The data to be clustered. Should be 1-dimensional.
        max_clusters (int): The maximum number of clusters to test.

    Returns:
        list: WCSS values for each number of clusters from 1 to max_clusters.
    """
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(
            n_clusters=i, 
            init='k-means++', 
            max_iter=300, 
            n_init=10, 
            random_state=42
        )
        kmeans.fit(sampled_data.reshape(-1, 1))
        wcss.append(kmeans.inertia_)
    return wcss

# Function to perform clustering, suppots KMeans, Mini-Batch KMeans and GMM
def perform_clustering(cluster_algo, data, sample_indices, original_image, mask):
    """
    Perform clustering on the provided data and map the results back to the original image.

    Args:
        cluster_algo (sklearn clustering object): The clustering algorithm to use (e.g., KMeans, MiniBatchKMeans, GaussianMixture).
        data (numpy.ndarray): The data to cluster, flattened.
        sample_indices (numpy.ndarray): Indices of the sampled data for training the clustering algorithm.
        original_image (numpy.ndarray): The original image data.
        mask (numpy.ndarray): Boolean mask indicating valid data points in the original image.
    
    Returns:
        numpy.ndarray: The labels of the clustered data mapped back to the original image dimensions.
    """
    # Train the clustering algorithm on the sampled data
    cluster_algo.fit(data[sample_indices].reshape(-1, 1))
    
    # Predict clusters for the full dataset
    labels = cluster_algo.predict(data.reshape(-1, 1))
    
    # Reshape labels to be placed back into the original image size
    full_labels = np.full(original_image.shape, -1)  # -1 for masked areas
    full_labels[mask] = labels  # Fill in the cluster labels in unmasked areas
    
    # Calculate cluster statistics
    unique_labels = np.unique(labels)
    stats = {label: {"min": float('inf'), "max": float('-inf'), "mean": 0, "std": 0} for label in unique_labels}
    
    for label in unique_labels:
        label_data = data[labels == label]
        stats[label]["min"] = np.min(label_data)
        stats[label]["max"] = np.max(label_data)
        stats[label]["mean"] = np.mean(label_data)
        stats[label]["std"] = np.std(label_data)
    
    print(f"Statistics for {cluster_algo.__class__.__name__}:")
    for label, stat in stats.items():
        print(f"Cluster {label}: Min={stat['min']}, Max={stat['max']}, Mean={stat['mean']}, Std Dev={stat['std']}")
    
    return full_labels

if __name__ == '__main__':
    # Load the data
    file_path = '/path/to/image/file'
    with rasterio.open(file_path) as src:
        image = src.read(1)  # Assuming a single band image
        meta = src.meta # Store original metadata
    # Mask out values in the range of 0-2 meters
    mask = (image > 2)
    masked_image = image[mask]

    # Flatten the masked image data for clustering
    data = masked_image.flatten()

    # Randomly sample 10% of the data for training
    np.random.seed(42)
    sample_indices = np.random.choice(len(data), size=int(0.1 * len(data)), replace=False)
    sampled_data = data[sample_indices]
    
    # PART 1) WCSS vs N plot for Elbow-Point Method
    max_clusters = 24
    wcss = calculate_wcss(sampled_data, max_clusters)

    # Plotting the WCSS curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method For Optimal k with KMeans')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.xticks(range(1, max_clusters + 1))  # Set x-ticks to show each cluster number
    plt.grid(True)
    plt.show()

    # Part 2) Clustering on the Image and displaying thei stats 
    # Define clustering models
    models = {
        "KMeans": KMeans(n_init=10, n_clusters=3, random_state=42),
        "MiniBatchKMeans": MiniBatchKMeans(n_clusters=3, random_state=42),
        "GaussianMixture": GaussianMixture(n_components=3, random_state=42),
    }

    # Custom color map for visualization
    from matplotlib.colors import ListedColormap
    viridis_mod = ListedColormap(np.vstack(([[0, 0, 0, 1]], plt.cm.viridis(np.linspace(0, 1, 256)))))
    bounds = [-1, 0, 1, 2, 3]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, viridis_mod.N)

    # Perform clustering and plot results for N = 3 Clusters with different algorithms
    for name, model in models.items():
        print(f"Processing {name}...")
        labels = perform_clustering(model, data, sample_indices, image, mask)

        plt.figure(figsize=(10, 8))
        plt.imshow(labels, cmap=viridis_mod, interpolation='none', norm=norm)
        plt.title(f"{name} Clustering with Excluded Data Marked")
        plt.colorbar()
        plt.show()
        # Save clustered labels as a new GeoTIFF
        output_file_path = f'/path/to/save/directory/{name}_BOMBAY.tif'
        new_meta = meta.copy()
        new_meta.update(dtype=rasterio.uint8, count=1, nodata = 255)
        
        with rasterio.open(output_file_path, 'w', **new_meta) as dst:
            dst.write(labels.astype(rasterio.uint8), 1)
        
        print(f"GeoTIFF saved to {output_file_path}")
