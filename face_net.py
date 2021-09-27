import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from model.FaceNet import Facenet as facenet

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和backbone需要修改！
#--------------------------------------------#
class Facenet(object):
    _defaults = {
        "model_path"    : "model/facenet_inception_resnetv1.pth",# optional: "model/facenet_mobilenet.pth" "model/facenet_inception_resnetv1.pth"
        "input_shape"   : (160, 160, 3),
        "backbone"      : "inception_resnetv1",#optional "mobilenet" "inception_resnetv1"
        "cuda"          : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Facenet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()
        
    def generate(self):
        # 载入模型
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = facenet(backbone=self.backbone, mode="predict")
        model.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net = model.eval()

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
            
        print('{} model loaded.'.format(self.model_path))
    
    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        if self.input_shape[-1]==1:
            new_image = new_image.convert("L")
        return new_image
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    
    def get_embedding(self,image):
        with torch.no_grad():
            #---------------------------------------------------#
            #   图片预处理，归一化
            #---------------------------------------------------#
            image = self.letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
            photo = torch.from_numpy(np.expand_dims(np.transpose(np.asarray(image).astype(np.float64)/255,(2,0,1)),0)).type(torch.FloatTensor)
            if self.cuda:
                photo = photo.cuda()
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            output = self.net(photo).cpu().numpy()
        return output
    
    def detect_image(self, image_1, image_2):

        output1 = self.get_embedding(image_1)
        output2 = self.get_embedding(image_2)
        #---------------------------------------------------#
        #   计算二者之间的距离
        #---------------------------------------------------#
        l1 = np.linalg.norm(output1-output2, axis=1)
        print("l1_distance_{}".format(l1))
        return l1
    