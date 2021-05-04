import shutil
import os
import argparse
import csv


def read_csv(source_path="./", target_path="./", csv_path="./", image_number=60000):
    numofimages = 0
    if not os.path.isdir(source_path):
        print('source_path ', source_path, ' is not a dir')
        return
    if not os.path.isdir(target_path):
        print('target_path ', target_path, ' is not a dir')
        return
    if not os.path.isfile(csv_path):
        print('csv_path ', csv_path, ' is not a file')
        return
    with open(csv_path, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        pid_index = 0
        img_index = 1
        for row in reader:
            if numofimages >= image_number:
                break
            img_list = row[img_index].split(' ')
            for img in img_list:
                if len(img) != 16:
                    continue
                dir_prefix = img[0] + "\\" + img[1] + "\\" + img[2]
                img_path = os.path.join(source_path, dir_prefix)
                if not os.path.isfile(os.path.join(img_path, img + ".jpg")) or row[pid_index] == '':
                    continue
                full_source_path = os.path.join(img_path, img + ".jpg")
                t_path = os.path.join(target_path, dir_prefix)
                if not os.path.isdir(t_path):
                    os.makedirs(t_path)
                full_target_path = os.path.join(t_path, img + ".jpg")
                shutil.copy(full_source_path, t_path)
                numofimages += 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, help="Path to find google landmark images")
    parser.add_argument("--target_path", type=str, help="Path to put the evaluation images")
    parser.add_argument("--csv_path", type=str, help="Path to clean csv file")
    parser.add_argument("--image_number", type=int, help="Number of evaluation images")
    args = parser.parse_args()
    return args


def main():
    print("start")
    # python split_google_landmarks.py --source_path D:/Datasets/GoogleLandmarks/train --target_path D:/Datasets/GoogleLandmarks/eval --csv_path D:/Datasets/GoogleLandmarks/train/train_clean.csv --image_number 60000
    config = parse_arguments()
    source_path = config.source_path
    target_path = config.target_path
    csv_path = config.csv_path
    image_number = config.image_number
    read_csv(source_path, target_path, csv_path, image_number)
    print("finished")


if __name__ == "__main__":
    main()