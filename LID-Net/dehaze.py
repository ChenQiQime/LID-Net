import torch
import torchvision
import torch.optim

import numpy as np
from PIL import Image
import glob
from network.net import UNet

def dehaze_image(image_path):
    data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(data_hazy) / 255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 1, 0)
    data_hazy = data_hazy.cuda().unsqueeze(0)
    print(data_hazy.shape)

    dehaze_net = UNet().cuda()
    dehaze_net.load_state_dict(torch.load('snapshots/best.pth'))

    clean_image = dehaze_net(data_hazy)
    clean_image = clean_image.permute(0, 1, 3, 2)
    data_hazy = data_hazy.permute(0, 1, 3, 2)
    torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "./results/" + image_path.split("\\")[-1])


if __name__ == '__main__':

    test_list = glob.glob(r"test/*")

    for image in test_list:
        dehaze_image(image)
        print(image, "done!")
