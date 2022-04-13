import cv2
import torch
from kmeans_torch import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import pdb
import pickle
from pykeops.torch import LazyTensor

# utility function to be used in plotting dense correspondences
def drawMatches(imageA, imageB, correspondences):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the matches
    for r, c, r_, c_ in correspondences:
        # draw the match
        ptA = c, r
        ptB = c_ + wA, r_
        cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        cv2.circle(vis, (c,r), radius=0, color=(0, 255, 0), thickness=4)
        cv2.circle(vis, (c_+wA,r_), radius=0, color=(0, 255, 0), thickness=4)
        
    return vis


def get_dense_correspondences_from_embeddings(embedding_map_1, embedding_map_2, acc_map_1, acc_map_2, img_1, img_2, acc_thresh=0.5, k=10, window_size=5, visualize=False):
    '''
    Find dense correspondences between frames at idx and idx+1,
    given embedding maps.
    embedding_map_1: (H, W, d)
    embedding_map_2: (H, W, d)
    acc_map_1: (H, W) -- used to find relevant FG pixels in first frame
    acc_map_2: (H, W) -- used to find relevant FG pixels in second frame
    '''
    FG_idxs = torch.where(acc_map_1 > acc_thresh)
    # sample `k` indices 
    idxs = torch.randperm(FG_idxs[0].shape[0])[:k]
    FG_idxs = (FG_idxs[0][idxs], FG_idxs[1][idxs])

    # Get dense correspondences
    dense_correspondences = []
    for i in range(FG_idxs[0].shape[0]):
        r, c = FG_idxs[0][i], FG_idxs[1][i]
        embedding = embedding_map_1[r, c]

        # search window
        min_dist = float('inf')
        min_r, min_c = -1, -1
        for r_ in range(r-window_size, r+window_size+1):
            for c_ in range(c-window_size, c+window_size+1):
                if r_ < 0 or r_ >= embedding_map_2.shape[0] or c_ < 0 or c_ >= embedding_map_2.shape[1]:
                    continue
                if acc_map_2[r_, c_] < acc_thresh:
                    continue
                embedding_ = embedding_map_2[r_, c_]
                dist = torch.norm(embedding - embedding_)
                if dist < min_dist:
                    min_dist = dist
                    min_r, min_c = r_, c_
        if min_r == -1 or min_c == -1:
            print("Correspondence for {} not found".format((r, c)))
            continue
        dense_correspondences.append((r, c, min_r, min_c))

    vis = drawMatches(img_1, img_2, dense_correspondences)
    if visualize:
        plt.imshow(vis)
        plt.axis('off')
        plt.show()
        plt.close()

    return dense_correspondences


def get_segmentation_from_embeddings(embedding_map, acc_map, img, acc_thresh=0.1, k=5, n_iter=10, visualize=False, save_path=None):
    '''
    Get segmentation from embedding map.
    embedding_map: (H, W, d)
    acc_map: (H, W) -- used to find relevant FG pixels in frame
    img: (H, W, 3) -- used to visualize segmentation
    '''
    # Get segmentation
    segmentation = torch.zeros(embedding_map.shape[0], embedding_map.shape[1], dtype=torch.uint8)
    segmentation[acc_map < acc_thresh] = k # background

    FG_embedding_idxs = torch.where(acc_map > acc_thresh)
    FG_embeddings = embedding_map[FG_embedding_idxs]

    cl, c = KMeans(FG_embeddings, k, Niter=n_iter)
    cl = cl.cpu()

    for i in range(FG_embedding_idxs[0].shape[0]):
        idx = (FG_embedding_idxs[0][i], FG_embedding_idxs[1][i])
        segmentation[idx] = cl[i]

    # create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # plot original image
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    # plot segmentation
    axes[1].imshow(segmentation.numpy(), cmap="tab10")
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    if visualize:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()
        
    return segmentation


def get_video_segmentation_from_embeddings(folder_path, acc_thresh=0.1, k=5, n_iter=10):
    embeddings = []
    # collect all FG embeddings
    for file in os.listdir(folder_path):
        if file.endswith('embed.pkl'):
            with open(os.path.join(folder_path, file), 'rb') as f:
                data = pickle.load(f)
            
            embedding_map = torch.tensor(data['embed_map'])
            acc_map = torch.tensor(data['acc_map'])
            FG_embeddings = embedding_map[torch.where(acc_map > acc_thresh)]
            embeddings.append(FG_embeddings)

    embeddings = torch.cat(embeddings, dim=0)

    # apply kmeans
    _, c = KMeans(embeddings, k, Niter=n_iter)
    c_j = LazyTensor(c.view(1, c.shape[0], c.shape[1]))

    # get segmentation map for each image
    for file in os.listdir(folder_path):
        if file.endswith('embed.pkl'):
            with open(os.path.join(folder_path, file), 'rb') as f:
                data = pickle.load(f)

            embedding_map = torch.tensor(data['embed_map'])
            acc_map = torch.tensor(data['acc_map'])

            fname, suffix = file.split('.')
            img_name = fname.split('_')[0]
            # img_path = os.path.join(folder_path, img_name + '.png')
            # img = imageio.imread(img_path)

            segmentation = torch.zeros(embedding_map.shape[0], embedding_map.shape[1], dtype=torch.uint8)
            segmentation[acc_map < acc_thresh] = k # background

            FG_embedding_idxs = torch.where(acc_map > acc_thresh)
            FG_embeddings = embedding_map[FG_embedding_idxs]
            N, D = FG_embeddings.shape
            x_i = LazyTensor(FG_embeddings.view(N, 1, D))

            D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
            cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

            for i in range(FG_embedding_idxs[0].shape[0]):
                idx = (FG_embedding_idxs[0][i], FG_embedding_idxs[1][i])
                segmentation[idx] = cl[i]

            # create figure with subplots
            fig, axes = plt.subplots(1, 1, figsize=(11, 6))
            # plot original image
            # axes[0].imshow(img)
            # axes[0].set_title('Original')
            # axes[0].axis('off')
            # plot segmentation
            axes.imshow(segmentation.numpy(), cmap="tab20")
            axes.set_title('Segmentation')
            axes.axis('off')
            plt.savefig(os.path.join(folder_path, img_name + '_seg.png'))
            plt.close()            
