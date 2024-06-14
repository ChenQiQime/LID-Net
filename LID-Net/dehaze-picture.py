import cv2
import torch
import torch.optim
from network.net import UNet

#import yuanlainet
import numpy as np
from PIL import Image


def image_Tensor2ndarray(image_tensor: torch.Tensor):
    """
    将tensor转化为cv2格式
    """
    assert (len(image_tensor.shape) == 4 and image_tensor.shape[0] == 1)
    # 复制一份
    image_tensor = image_tensor.clone().detach()
    # 到cpu
    image_tensor = image_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    image_tensor = image_tensor.squeeze()
    # 从[0,results]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    image_cv2 = image_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    # image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    return image_cv2

def dehaze_image(image_path):
    #data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(image_path) / 255.0)

    # data_hazy=cv2.imread(image_path)
    # print(type(data_hazy))
    #data_hazy = image_path / 255.0
    data_hazy = torch.from_numpy(data_hazy).float()
    print(data_hazy.shape)
    data_hazy = data_hazy.permute(2, 1, 0)
    data_hazy = data_hazy.cuda().unsqueeze(0)

    dehaze_net = UNet().cuda()
    dehaze_net.load_state_dict(torch.load('snapshots/best.pth'))

    clean_image = dehaze_net(data_hazy)
    clean_image = clean_image.permute(0,1,3,2)


    clean_image=image_Tensor2ndarray(clean_image)


    return clean_image

    #torchvision.utils.save_image(torch.cat((data_hazy, clean_image), 0), "results/" + image_path.split("/")[-results])





frame=cv2.imread('NYU2_1_1_2.jpg')
frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
frame=Image.fromarray(np.uint8(frame))

#clean_image= dehaze_image(frame)
frame= dehaze_image(frame)
frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    # 显示结果帧e
#cv2.imshow('frame',frame)
cv2.imwrite('test.png', frame)
cv2.waitKey(1)

# 完成所有操作后，释放捕获器

cv2.destroyAllWindows()

# myPath = 'RTTS'
# # 输出目录
# outPath = r'C:/Users/Dr. Tao/Desktop/aod/dehaze-RTTS/'
# def processImage(filesource, destsource, name, imgtype):
#     '''
#     filesource是存放待雾化图片的目录
#     destsource是存放物化后图片的目录
#     name是文件名
#     imgtype是文件类型
#     '''
#     imgtype = 'jpeg' if imgtype == '.jpg' else 'png'
#     # 打开图片
#     frame = cv2.imread(name)
#     frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     frame=Image.fromarray(np.uint8(frame))
#
#     #clean_image= dehaze_image(frame)
#     frame= dehaze_image(frame)
#     frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
#
#     cv2.imwrite(destsource + name, frame)
#
#
# def run():
#     # 切换到源目录，遍历目录下所有图片
#     os.chdir(myPath)
#     for i in os.listdir(os.getcwd()):
#         # 检查后缀
#         postfix = os.path.splitext(i)[results]
#         print(postfix, i)
#         if postfix == '.jpg' or postfix == '.png':
#             processImage(myPath, outPath, i, postfix)
#
#
# if __name__ == '__main__':
#
#     run()
