import os

path_segm = '../_out/lidarSegm/'

def delete_pcls():
    for f in os.listdir(path_segm):
        os.remove(os.path.join(path_segm, f))

if __name__ == '__main__':
    delete_pcls()