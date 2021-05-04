from torchreid.data import ImageDataset
from torchreid.utils import read_image
import os
from PIL import Image
import random
import csv
import copy

#Class for preprocessing image data for training
class LandmarksDataset(ImageDataset):

    def __init__(self, root="", **kwargs):
        # important: all labels that appear in query/gallery have to appear in the train dataset too
        self.label_mapping = dict()
        self.img_label_list = dict()
        train, query, gallery = self.create_img_list(root, mode="train")
        super(LandmarksDataset, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):
        img_path, pid, camid, dsetid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid
        }

        return item

    def create_img_list(self, path="./", part_dataset=True, mode="train"):
        minimal_number_of_train_images_per_label = 2
        dir = os.path.join(path, mode)
        if mode == "train":
            file_name = "train_clean.csv"
        img_index = 1 if mode == "train" else 0  # 0 for index set
        pid_index = 0 if mode == "train" else 1  # 1 for index set

        with open(os.path.join(dir, file_name), newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            train_path_list = []
            train_pid_list = []
            query_path_list = []
            query_pid_list = []
            test_path_list = []
            test_pid_list = []
            counter = 0
            for row in reader:
                img_list = row[img_index].split(' ')
                number = 0
                for img in img_list:
                    if len(img) != 16:
                        continue
                    dir_prefix = img[0] + "/" + img[1] + "/" + img[2]
                    img_path = os.path.join(dir, dir_prefix)
                    if not os.path.isfile(os.path.join(img_path, img + ".jpg")) or row[pid_index] == '':
                        continue
                    number += 1
                if number < minimal_number_of_train_images_per_label + 2:
                    continue
                if part_dataset:
                    b1 = False
                    b2 = False
                else:
                    b1 = True
                    b2 = True
                for img in img_list:
                    if len(img) != 16:
                        continue
                    dir_prefix = img[0] + "/" + img[1] + "/" + img[2]
                    img_path = os.path.join(dir, dir_prefix)
                    if not os.path.isfile(os.path.join(img_path, img + ".jpg")) or row[pid_index] == '':
                        continue
                    if not b1:
                        test_pid_list.append(counter)  # int(row[pid_index]))
                        test_path_list.append(os.path.join(img_path, img + ".jpg"))
                        b1 = True
                    else:
                        if not b2:
                            query_pid_list.append(counter)  # int(row[pid_index]))
                            query_path_list.append(os.path.join(img_path, img + ".jpg"))
                            b2 = True
                        else:
                            train_pid_list.append(counter)  # int(row[pid_index]))
                            train_path_list.append(os.path.join(img_path, img + ".jpg"))

                counter += 1
            if part_dataset:
                # image_path, label, camera_id
                train_list = list(zip(train_path_list, train_pid_list, [0] * len(train_path_list)))
                query_list = list(zip(query_path_list, query_pid_list, [1] * len(query_path_list)))
                test_list = list(zip(test_path_list, test_pid_list, [2] * len(test_path_list)))

                random.shuffle(train_list)
                random.shuffle(query_list)
                random.shuffle(test_list)
                return train_list, query_list, test_list

    def remove_from_list(self, item, list):
        list_copy = copy.deepcopy(list)
        for i in list:
            if (item[0] == i[0] and item[1] == i[1] and item[2] == i[2]):
                list_copy.remove(item)
        return list_copy

    def show_summary(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print(
            '  train    | {:5d} | {:8d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:8d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:8d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  ----------------------------------------')
