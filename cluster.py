# ================================= Imports =================================

import os
import random
import shutil
import warnings

import cv2
import imutils
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tqdm import tqdm
warnings.filterwarnings('ignore')

# =============================== Cluster Class ===============================

class Cluster:
    '''
    Create Cluster object
    '''
    def __init__(self, img_size=(224, 224), model_path=''):
        """ 
        Initializes Cluster object
    
        Sets image size and model to use for feature embeddings extraction
    
        Parameters: 
        img_size (tuple(int,int)): default-(224,224), size of image as tuple of (width, height)
        model_path (string): default-'', ResNet50 if '' otherwise path to custom model directory
        """
        self.img_size = img_size
        self.model_path = model_path
        self.images = pd.Series()
        if self.model_path:
            loaded_json = open(self.model_path + '.json', 'r').read()
            model = model_from_json(loaded_json)
            model.load_weights(self.model_path + '.h5')
            outputs = model.layers[-1].output
        else:
            model = ResNet50(weights='imagenet')
            outputs = model.get_layer('avg_pool').output

        self.model = Model(inputs=model.input, outputs=outputs)

    def vec(self, img_path):
        """ 
        Returns feature embeddings of an Image
    
        Uses model avg_pool layer if default(ResNet50) is used or layers[-1] if custom model is used to extract feature embeddings.
    
        Parameters:
        img_path (string): path of image

        Returns:
        feature embeddings of image
        """
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        if not self.model_path:
            x = preprocess_input(x)
        else:
            x = x / 255.0
        return self.model.predict(x)[0]

    def load_images(self, path):
        """ 
        Loads images and stores images' embeddings
    
        loads images for path provided and stores embeddings using <function vec>
    
        Parameters:
        path (string/pd.Series): path to images directory as string or pandas series with all images path
        """
        if type(path) == str:
            self.images = pd.Series([os.path.join(path, img) for img in os.listdir(path)])
        elif type(path) == pd.Series:
            self.images = path.reset_index(drop=True)
        else:
            print ('Image paths not supported! Provide either <path_to_images_dir> or a <pandas Series> with image paths')
            return

        self.embeddings = np.array([self.vec(img_path) for img_path in tqdm(self.images.values)])

    def find_best_k(self, plot=True):
        """ 
        Finds optimal n_clusters
    
        Uses silhouette score to identify optimal n_clusters only if <function load_images> is used earlier.
    
        Parameters:
        plot (boolean): default-True, To visualize silhouette score for n_cluster or not
        """
        if len(self.images.index)==0:
            print ('Images/Embeddings not Found! Use load_images method to load and extract feature embeddings of images.')
            return

        s_scores = []
        max_k = int(np.sqrt(self.images.shape[0])) + 1
        self.kmeans_labels = dict()

        for n in range(2, max_k):
            kmeans = KMeans(n_clusters=n, random_state=0, n_jobs=-1).fit(self.embeddings)
            self.kmeans_labels[n] = kmeans.labels_
            score = silhouette_score(self.embeddings, kmeans.labels_)
            s_scores.append(score)

        self.best_k = np.array(s_scores).argmax() + 2

        if plot:
            plt.plot(range(2, max_k), s_scores, color='g', linewidth='2')
            plt.xlabel('No of Clusters')
            plt.xticks(np.arange(2, max_k, 1.0))
            plt.ylabel('Silhouette scores')
            plt.show()

    def show_clusters(self, labels):
        tsne = TSNE(n_components=2, random_state=0)
        reduced = tsne.fit_transform(self.embeddings)
        palette = sns.color_palette('hls', n_colors=len(set(labels)))
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette=palette)
        plt.axis('off')
        plt.show()

    def run_kmeans(self, n_clusters=0):
        """ 
        Make clusters
    
        Uses optimal n_clusters set by <function find_best_k> or using custom n_cluster to visualize and run further analysis
    
        Parameters:
        n_clusters (int): default-0, if default optimal n is used otherwise custom n is used
        """
        if n_clusters == 0:
            n_clusters = self.best_k

        if n_clusters in self.kmeans_labels:
            labels = self.kmeans_labels[n_clusters]
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=4).fit(self.embeddings)
            labels = kmeans.labels_

        self.show_clusters(labels)

        self.cluster_indices = {}
        for n in range(n_clusters):
            self.cluster_indices[n] = list(np.argwhere(labels == n).flatten())

        for (cluster, indices) in sorted(self.cluster_indices.items(), key=lambda x: x[0]):
            total = len(indices)
            print(f'CLUSTER: {cluster} --> {total} images')

            random_indices = random.sample(indices, min(10, total))

            image_set = []
            for path in self.images.loc[random_indices]:
                image_set.append(image.load_img(path, target_size=self.img_size))

            plt.figure(figsize=(12, 7))
            plt.imshow(np.hstack(image_set))
            plt.axis('off')
            plt.show()

    def get_cluster(self, n):
        """ 
        Visualize a Cluster
    
        visualize a specific cluster using a cluster montage
    
        Parameters:
        n (int): cluster to visualize
        """
        image_set = []
        for path in self.images.loc[self.cluster_indices[n]]:
            image_set.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

        row = int(np.ceil(len(image_set) / 10))
        col = min(10, len(image_set))

        montages = imutils.build_montages(image_set, self.img_size, (col, row))
        for montage in montages:
            plt.figure(figsize=(20, row * 5))
            plt.imshow(montage)
            plt.axis('off')
            plt.show()

    def extract_cluster(self, n, extraction_type='copy', to='ClusterCuts' ):
        """ 
        Cut/Copy images of a cluster
    
        extract a cluster using copy or move method to a directory
    
        Parameters:
        n (int): cluster to extract
        extraction_type (string): either 'copy' cluster images or 'cut' from current directory
        to (string): path of directory which will store extracted images
        """
        for img_path in self.images.loc[self.cluster_indices[n]]:
            image_name = os.path.basename(img_path)

            if not os.path.exists(to):
                os.mkdir(to)

            if extraction_type == 'copy':
                shutil.copyfile(img_path, os.path.join(to, image_name))
            else:
                shutil.move(img_path, os.path.join(to, image_name))


# =============================================================================