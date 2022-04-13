import torch
import pickle
import numpy as np
import os
import imageio
from tqdm import tqdm

from kmeans_torch import KMeans
from embed_utils import get_segmentation_from_embeddings, get_video_segmentation_from_embeddings

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    
    parser.add_argument("--folder_path", type=str, 
                        help='path of folder with pickle files')
    parser.add_argument("--k", type=int, default=5, 
                        help='number of clusters in KMeans')
    parser.add_argument("--n_iter", type=int, default=10,
                        help='num iterations in KMeans')
    parser.add_argument("--acc_thresh", type=float, default=0.5)
    parser.add_argument("--full_video", action='store_true', 
                        help='Compute segmentation for full video at once')
    
    return parser

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    
    if not args.full_video:
        # iterate over all .pkl files in args.folder_path
        for file in os.listdir(args.folder_path):
            if file.endswith('embed.pkl'):
                with open(os.path.join(args.folder_path, file), 'rb') as f:
                    data = pickle.load(f)
                
                fname, suffix = file.split('.')
                img_name = fname.split('_')[0]
                img_path = os.path.join(args.folder_path, img_name + '.png')
                img = imageio.imread(img_path)
                
                try:
                    segmentation = get_segmentation_from_embeddings(torch.tensor(data['embed_map']), torch.tensor(data['acc_map']), img,
                                                acc_thresh=args.acc_thresh, k=args.k, n_iter=args.n_iter,
                                                save_path=os.path.join(args.folder_path, img_name + '_seg.png'))
                except:
                    print("Could not get segmentation for {}".format(img_name))
                    continue
    
    else:
        get_video_segmentation_from_embeddings(args.folder_path, acc_thresh=args.acc_thresh, k=args.k, n_iter=args.n_iter)