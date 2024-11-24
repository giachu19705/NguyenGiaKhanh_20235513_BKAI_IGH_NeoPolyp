import torch
from torch import nn
import argparse
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Dropout(p=0.3),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv(x)

class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_block, self).__init__()
        self.conv = conv_block(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        merge_layer = self.conv(x)
        next_layer = self.max_pool(merge_layer)
        return next_layer, merge_layer

class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv = conv_block(out_channels * 2, out_channels)
        
    def forward(self, x, merge_layer):
        x = self.up_conv(x)
        x = torch.cat([x, merge_layer], axis = 1)
        next_layer = self.conv(x)
        return next_layer

class UNet(nn.Module):
    def __init__(self, n_class=3):
        super(UNet, self).__init__()
        # Encoder blocks
        self.enc1 = encoder_block(3, 64)
        self.enc2 = encoder_block(64, 128)
        self.enc3 = encoder_block(128, 256)
        self.enc4 = encoder_block(256, 512)
        
        # Bridge_blocks
        self.bridge = conv_block (512, 1024)
        # Decoder blocks
        self.dec1 = decoder_block(1024, 512)
        self.dec2 = decoder_block(512, 256)
        self.dec3 = decoder_block(256, 128)
        self.dec4 = decoder_block(128, 64)
        
        # 1x1 convolution
        self.out = nn.Conv2d(64, n_class, 1, 1)
        
    def forward(self, image):
        n1, s1 = self.enc1(image)
        n2, s2 = self.enc2(n1)
        n3, s3 = self.enc3(n2)
        n4, s4 = self.enc4(n3)
        
        n5 = self.bridge(n4)
        
        n6 = self.dec1(n5, s4)
        n7 = self.dec2(n6, s3)
        n8 = self.dec3(n7, s2)
        n9 = self.dec4(n8, s1)
        
        output = self.out(n9)
        
        return output
    
model = UNet()
path = './unet_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model, path):
    checkpoint = torch.load(path, map_location=device)
    new_state_dict = {}
    for key, value in checkpoint['model'].items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value  # Bỏ 'module.' ở đầu
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    return model

model = load_model(model, path)

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

color_dict= {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0)}
def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))

    for k in color_dict.keys():
        output[mask==k] = color_dict[k]

    return np.uint8(output)   

def Test(img_path):
    trainsize = 256
    model.eval()
    
    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_w = ori_img.shape[0]
    ori_h = ori_img.shape[1]
    img = cv2.resize(ori_img, (trainsize, trainsize))
    transformed = val_transform(image=img)
    input_img = transformed["image"]
    input_img = input_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output_mask = model.forward(input_img).squeeze(0).cpu().numpy().transpose(1,2,0)
    mask = cv2.resize(output_mask, (ori_h, ori_w))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask, color_dict)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./segmented_image.jpeg", mask_rgb)


def main():
    parser = argparse.ArgumentParser(description="Infer script")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    image_path = args.image_path

    Test(image_path)
   
if __name__ == "__main__":
    main()