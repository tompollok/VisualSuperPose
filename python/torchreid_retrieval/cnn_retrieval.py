import time
import glob
from torchreid.utils import FeatureExtractor
import torch
import os
import torchreid
import cv2
import argparse
from tqdm import tqdm

#use cnn for feature extraction
def feature_extraction(query_path, gallery_path, model_name, model_path, results_path="./log/results/", num_imgs=5):
    extractor = FeatureExtractor(model_name=model_name,
                                 model_path=model_path,
                                 device="cuda",
                                 pixel_norm=True)

    features_query = extractor(query_path)

    distances = []
    counter = 0
    start = time.time()
    gallery_list = list(glob.iglob(f'{gallery_path}/**/*.jpg', recursive=True))

    with tqdm(total=len(gallery_list)) as searching_bar:
        for img in gallery_list:
            if img.endswith("jpg") or img.endswith("png") or img.endswith("jpeg"):
                try:
                    features_gallery = extractor(os.path.join(gallery_path, img))
                    equal = torch.sum(torch.eq(features_gallery, features_query))
                except IOError:
                    print("File not processsed", gallery_path, img)
                dist = torchreid.metrics.distance.euclidean_squared_distance(features_gallery, features_query)
                distances.append((os.path.join(gallery_path, img), dist))
                counter += 1
                # print("img: ", gallery_path, img)
                searching_bar.update(1)

    end = time.time()

    print("time: ", end - start)
    print("counter: ", counter)
    distances = sorted(distances, key=lambda x: x[1])

    distances = distances[:num_imgs]

    qImg = cv2.imread(query_path)

    if not os.path.isdir(results_path):
        os.chdir("./log")
        os.mkdir("results")
        results_path = "./log/results/"
        os.chdir("../")

    cv2.imwrite(os.path.join(f"{results_path}", "query.jpg"), qImg)

    with open(os.path.join(results_path, 'path.txt'), 'w') as f:
        for idx, pair in enumerate(distances):
            retrieved = cv2.imread(pair[0])
            cv2.imwrite(os.path.join(f"{results_path}", f"retrieval_{idx}.jpg"), retrieved)
            f.write(os.path.split(pair[0])[-1]+'\n')
            print("retrieved:", pair[0])


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50", help="Model architecture name")
    parser.add_argument("--model_weights", type=str, help="Path to .pth.tar file")
    parser.add_argument("--query_img", type=str, help="Path to query image")
    parser.add_argument("--gallery", type=str, help="Path to gallery directory")
    parser.add_argument("--results_path", type=str, help="Path to save result images to")
    parser.add_argument("--retrieved_images", type=int, help="Number of images to retrieve")
    args = parser.parse_args()
    return args


def main():
    # example usage:
    'python cnn_retrieval.py --model_name resnet50 --model_weights /home/johanna/Documents/Bildauswertung/cnn-models/resnet50_150k/model.pth.tar-50 --query_img /media/johanna/Volume/colmap_images/Gendarmenmarkt/home/wilsonkl/projects/SfM_Init/dataset_images/Gendarmenmarkt/11648097_826b97c4f6_o.jpg --gallery /media/johanna/Volume/colmap_images/ --results_path ./log/results --retrieved_images 20'

    config = parse_arguments()
    print(config)
    feature_extraction(config.query_img, config.gallery, config.model_name, config.model_weights, config.results_path)

    # To use this script without command line arguments, uncomment the following lines:
    '''
    model_name = "resnet50"
    model_weights = "/home/johanna/Documents/Bildauswertung/cnn-models/resnet50_150k/model.pth.tar-50"
    query_img = "/media/johanna/Volume/colmap_images/Gendarmenmarkt/home/wilsonkl/projects/SfM_Init/dataset_images/Gendarmenmarkt/11648097_826b97c4f6_o.jpg"
    gallery = "/media/johanna/Volume/colmap_images/"
    results_path = "./log/results"

    feature_extraction(query_img,
                       gallery,
                       model_name,
                       model_weights,
                       results_path,
                       20)
   '''


if __name__ == "__main__":
    main()
