import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random


class UnalignedScaleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # if opt.split == "":
        if opt.phase == "train":
            self.dir_A = os.path.join(opt.dataroot, "train/A")
            self.dir_B = os.path.join(opt.dataroot, "train/B")
        if opt.phase == "test":
            self.dir_A = os.path.join(opt.dataroot, "val/A")
            self.dir_B = os.path.join(opt.dataroot, "val/B")
        # else:
        #     self.dir_A = os.path.join(opt.dataroot, opt.split, "A")
        #     self.dir_B = os.path.join(opt.dataroot, opt.split, "B")

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        # self.transform = get_transform(opt)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))

        # read the triplet from A and B --
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # A = self.transform(A_img)
        # B = self.transform(B_img)
        # get the triplet from A
        if self.opt.resize_mode == "scale_shortest":
            w, h = A_img.size
            if w >= h: 
                scale = self.opt.loadSize / h
                new_w = int(w * scale)
                new_h = self.opt.loadSize
            else:
                scale = self.opt.loadSize / w
                new_w = self.opt.loadSize
                new_h = int(h * scale)
        # 从BICUBIC修改至LANCZOS
            A_img = A_img.resize((new_w, new_h), Image.LANCZOS)
        elif self.opt.resize_mode == "square":
            A_img = A_img.resize((self.opt.loadSize, self.opt.loadSize), Image.LANCZOS)
        elif self.opt.resize_mode == "rectangle":
            A_img = A_img.resize((self.opt.loadSizeW, self.opt.loadSizeH), Image.LANCZOS)
        elif self.opt.resize_mode == "none":
            pass
        else:
            raise ValueError("Invalid resize mode!")

        A_img = self.transform(A_img)

        w = A_img.size(2)
        h = A_img.size(1)
        if self.opt.crop_mode == "square":
            fineSizeW, fineSizeH = self.opt.fineSize, self.opt.fineSize
        elif self.opt.crop_mode == "rectangle":
            fineSizeW, fineSizeH = self.opt.fineSizeW, self.opt.fineSizeH
        elif self.opt.crop_mode == "none":
            fineSizeW, fineSizeH = w, h
        else:
            raise ValueError("Invalid crop mode!")

        w_offset = random.randint(0, max(0, w - fineSizeW - 1))
        h_offset = random.randint(0, max(0, h - fineSizeH - 1))

        A_img = A_img[:, h_offset:h_offset + fineSizeH, w_offset:w_offset + fineSizeW]


        if self.opt.resize_mode == "scale_shortest":
            w, h = B_img.size
            if w >= h: 
                scale = self.opt.loadSize / h
                new_w = int(w * scale)
                new_h = self.opt.loadSize
            else:
                scale = self.opt.loadSize / w
                new_w = self.opt.loadSize
                new_h = int(h * scale)
                
            B_img = B_img.resize((new_w, new_h), Image.LANCZOS)
        elif self.opt.resize_mode == "square":
            B_img = B_img.resize((self.opt.loadSize, self.opt.loadSize), Image.LANCZOS)
        elif self.opt.resize_mode == "rectangle":
            B_img = B_img.resize((self.opt.loadSizeW, self.opt.loadSizeH), Image.LANCZOS)
        elif self.opt.resize_mode == "none":
            pass
        else:
            raise ValueError("Invalid resize mode!")

        B_img = self.transform(B_img)

        w = B_img.size(2)
        h = B_img.size(1)
        if self.opt.crop_mode == "square":
            fineSizeW, fineSizeH = self.opt.fineSize, self.opt.fineSize
        elif self.opt.crop_mode == "rectangle":
            fineSizeW, fineSizeH = self.opt.fineSizeW, self.opt.fineSizeH
        elif self.opt.crop_mode == "none":
            fineSizeW, fineSizeH = w, h
        else:
            raise ValueError("Invalid crop mode!")
        w_offset = random.randint(0, max(0, w - fineSizeW - 1))
        h_offset = random.randint(0, max(0, h - fineSizeH - 1))

        B_img = B_img[:, h_offset:h_offset + fineSizeH,
             w_offset:w_offset + fineSizeW]

        #######
        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc

        # if input_nc == 1:  # RGB to gray
        #    tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        #    A = tmp.unsqueeze(0)

        # if output_nc == 1:  # RGB to gray
        #    tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        #    B = tmp.unsqueeze(0)
        return {'A1': A_img, 'B1': B_img, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
