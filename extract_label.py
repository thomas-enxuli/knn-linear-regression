import struct
import numpy as np
from tqdm import tqdm
import os

root_dir = '/Users/thomas/Downloads/data_3d_semantics'
# pcdFile = '/Users/thomas/Downloads/data_3d_semantics/2013_05_28_drive_0000_sync/static/000002_000385.ply'


fmt = '=fffBBBiiB'
fmt_len = 24




def init_dict():
    # counts = {}
    # label_idx = list(range(45)) + [-1]
    #
    # for i in label_idx:
    #     counts[i] = 0
    # return counts
    num_classes = 46
    return [0]*num_classes

def readPly(pcdFile):

    pt_dict = init_dict()

    with open(pcdFile, 'rb') as f:
        plyData = f.readlines()

    headLine = plyData.index(b'end_header\n')+1
    plyData = plyData[headLine:]
    plyData = b"".join(plyData)

    n_pts_loaded = len(plyData) // fmt_len

    assert ((len(plyData) % fmt_len)==0)
    # assert(n_pts_loaded==n_pts)
    # n_pts_loaded = int(n_pts_loaded)
    print(str(n_pts_loaded)+' points loaded')

    data = []
    err = 0
    for i in range(n_pts_loaded):
        pts=struct.unpack(fmt, plyData[i*fmt_len:(i+1)*fmt_len])
        data.append(pts)

        # pts[6] semantic label
        # pts[7] instance label
        idx = int(pts[6])
        pt_dict[idx] += 1

    data=np.asarray(data)
    # print(err)

    # print(data[:5])
    # print(pt_dict)
    return data, pt_dict, n_pts_loaded

if __name__=='__main__':
    seq_dirs = [folder for folder in os.listdir(root_dir) if folder[0]!='.']
    for folder in seq_dirs:
        full_path = os.path.join(root_dir, folder, 'static')
        label_list = [file for file in os.listdir(full_path) if file[0]!='.']
        label_dict = init_dict()
        total_pts = 0
        bar = tqdm(total=len(label_list))
        for label_name in label_list:
            bar.update(1)
            full_label_path = os.path.join(full_path, label_name)
            _, cur_label_cnt, cur_pts = readPly(full_label_path)
            label_dict = [sum(x) for x in zip(label_dict, cur_label_cnt)]
            total_pts += cur_pts
        print(folder)
        print(label_dict)
        print(total_pts)
