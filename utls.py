
import numpy as np
import os
import cv2 as cv
def image_normalization(img, img_min=0, img_max=255):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255
    :return: a normalized image, if max is 255 the dtype is uint8
    """
    img = np.float32(img)
    epsilon=1e-12 # whenever an inconsistent image
    img = (img-np.min(img))*(img_max-img_min)/((np.max(img)-np.min(img))+epsilon)+img_min
    return img

def make_dirs(paths): # make path or paths dirs
    if not os.path.exists(paths):
        os.makedirs(paths)
        print("Directories have been created: ",paths)
        return True
    else:
        print("Directories already exists: ", paths)
        return False

def read_files_list(list_path,dataset_name=None):
    mfiles = open(list_path)
    file_names = mfiles.readlines()
    mfiles.close()

    file_names = [f.strip() for f in file_names] # this is for delete '\n'
    return file_names

def data_parser(dataset_dir,dataset_name, list_name=None,is_train=True):

    train_files_name = list_name # dataset base dir

    base_dir = os.path.join(dataset_dir,
                            dataset_name,'omsiv4colorization')\
        if dataset_name.upper()=='OMSIV' else os.path.join(dataset_dir,dataset_name) # or SSMIHD

    train_list_path = os.path.join(base_dir, train_files_name)
    data_list = read_files_list(train_list_path)
    tmp_list = data_list[0]
    tmp_list = tmp_list.split(' ')
    n_lists = len(tmp_list)
    if n_lists==1:
        data_list = [c.split(' ') for c in data_list]
        data_list = [(os.path.join(base_dir, c[0])) for c in data_list]
    elif n_lists==2:
        data_list = [c.split(' ') for c in data_list]
        data_list = [(os.path.join(base_dir, c[0]),
                       os.path.join(base_dir, c[1])) for c in data_list]
    else:
        raise NotImplementedError('there are just to entry files')

    num_data = len(data_list)
    print(" Enterely training set-up from {}, size: {}".format(train_list_path, num_data))

    all_train_ids = np.arange(num_data)
    np.random.shuffle(all_train_ids)
    if is_train:

        train_ids = all_train_ids[:int(0.9 * len(data_list))]
        valid_ids = all_train_ids[int(0.9 * len(data_list)):]

        print("Training set-up from {}, size: {}".format(train_list_path, len(train_ids)))
        print("Validation set-up from {}, size: {}".format(train_list_path, len(valid_ids)))
        cache_info = {
            "files_path": data_list,
            "train_paths":[data_list[i] for i in train_ids],
            "val_paths": [data_list[i] for i in valid_ids],
            "n_files": num_data,
            "train_indices": train_ids,
            "val_indices": valid_ids
        }
    else:
        tmp_img = cv.imread(data_list[0][0])
        tmp_nir = cv.imread(data_list[0][1])

        print("Testing set-up from {}, size: {}".format(train_list_path, len(all_train_ids)))
        cache_info = {
            "files_path": data_list,
            "n_files": num_data,
            "train_indices": all_train_ids,
            #               rgb shape and nir shape
            "data_shape":[tmp_img.shape,tmp_nir.shape]
        }

    return cache_info