from feature_extractors import CNNFeatureExtractor
from data_handling import load_images_from_path, plot_clusters, show_cluster_images
from dimensionality_reduction import PCAReduction
from clustering_algorithms import DBSCANClustering, KMeansClustering
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # Construct feature extractor
    feature_extractor = CNNFeatureExtractor((224, 224, 3))

    # Load Images
    image_batch = load_images_from_path(r"A:\Arbeit\Github\proj-feature-extraction\data\test_images")
    image_batch = feature_extractor.preprocess_images(image_batch)
    features = feature_extractor.extract_features(image_batch)

    # Reduce Dimensionality
    dim_reduction = PCAReduction(dims=3)
    reduced_features = dim_reduction.fit_to_data(features)

    # Cluster
    clustering = KMeansClustering("auto") #DBSCANClustering(eps=0.8) #
    labels = clustering.fit_to_data(reduced_features)

    # Plot
    plot_clusters(reduced_features, labels)
    show_cluster_images(image_batch, labels)