from torch.utils.data import Dataset as PytorchDataset
import numpy as np
import cv2
import os
import torch
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class KittiDataset(PytorchDataset):
    def __init__(
        self,
        imgs_left_dir,
        return_left_and_right=False,
        imgs_right_dir=None,
        max_num_imgs=None,
    ):

        self.width = 640
        self.height = 640

        self.max_num_imgs = max_num_imgs

        self.return_left_and_right = return_left_and_right
        self.imgs_left_dir = imgs_left_dir
        self.imgs_right_dir = imgs_right_dir
        self.imgs_filenames = sorted(os.listdir(self.imgs_left_dir))
        self.imgs_filenames = self.imgs_filenames[: self.max_num_imgs]

        self.imgs_seq_ids = np.array(
            [int(img_fn.split("_")[0]) for img_fn in self.imgs_filenames]
        )
        self.seq_ids_unique, self.imgs_seq_ids = np.unique(
            self.imgs_seq_ids, return_inverse=True
        )
        self.seq_ids_unique = np.unique(self.imgs_seq_ids)
        self.imgs_el_ids = np.array(
            [int(img_fn.split("_")[1].split(".")[0]) for img_fn in self.imgs_filenames]
        )
        self.el_ids_unique, self.imgs_el_ids = np.unique(
            self.imgs_el_ids, return_inverse=True
        )
        self.el_ids_unique = np.unique(self.imgs_el_ids)
        self.num_seq = len(self.seq_ids_unique)

        self.imgs_seq_lengths = np.array(
            [
                np.sum(np.array(self.imgs_seq_ids) == seq_id)
                for seq_id in self.seq_ids_unique
            ]
        )
        self.imgs_seq_lengths_acc = np.add.accumulate(
            np.insert(self.imgs_seq_lengths, 0, values=0)
        )

        self.imgspairs_seq_lengths = self.imgs_seq_lengths - 1
        self.imgspairs_seq_lengths_acc = np.add.accumulate(
            np.insert(self.imgspairs_seq_lengths, 0, values=0)
        )

        self.imgpairs_seq_ids = np.delete(
            self.imgs_seq_ids, self.imgs_seq_lengths_acc[:-1]
        )
        self.imgpairs_el_ids = np.delete(
            self.imgs_el_ids, self.imgs_seq_lengths_acc[1:] - 1
        )

        self.num_imgpairs = len(self.imgpairs_seq_ids)
        print("self.num_seq", self.num_seq)
        print("self.num_imgpairs", self.num_imgpairs)

    def __getitem__(self, imgpair_id):
        imgpair_seq_id = self.imgpairs_seq_ids[imgpair_id]
        imgpair_el_id = self.imgpairs_el_ids[imgpair_id]

        # return imgpair_seq_id
        return imgpair_seq_id * 20 + imgpair_el_id

    def __len__(self):
        return self.num_imgpairs


train_dataset_max_num_imgs = None
val_dataset_max_num_imgs = None
num_imgpairs_forward = 1
datasets_dir = "../../optical-flow/datasets"


train_dataset = KittiDataset(
    imgs_left_dir=os.path.join(datasets_dir, "KITTI_flow_multiview/testing/image_2"),
    max_num_imgs=train_dataset_max_num_imgs,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=num_imgpairs_forward, shuffle=True
)

import collections


class DataloaderIterator(collections.Iterator):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __next__(self):
        try:
            batch = self.iterator.next()
        except:
            print("end of iter reached - restat")
            self.iterator = iter(self.dataloader)
            batch = next(self)
            print("batch", batch)
        return batch

    def __len__(self):
        return len(self.iterator)


train_it = DataloaderIterator(train_dataloader)
print("len(train_it)", len(train_it))
epoch_length = 1000
seq_ids = []
el_ids = []

total_it = 0
for j in range(10):
    for iteration, batch in enumerate(train_it):
        total_it += 1
        # batch = next(train_it)

        seq_ids.append(batch[0].numpy())
        if iteration == epoch_length - 1:
            break
    print("len(train_it)", len(train_it))
    print("Training iteration", total_it, batch[0].numpy())
    plt.hist(seq_ids, bins=3989)
    plt.show()
    cv2.waitKey(0)

# noc: flow only for non-occluded
# occ: flow for all pixels
val_dataset = KittiDataset(
    imgs_left_dir=os.path.join(datasets_dir, "KITTI_flow/training/image_2"),
    max_num_imgs=val_dataset_max_num_imgs,
)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
