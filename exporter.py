from genericpath import exists
from numpy.lib.function_base import insert
import torch
from torchvision import transforms,datasets
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil
import os
import networks
from layers import disp_to_depth
import struct
import pandas as pd

outputs = {}

def hook(module,input,output):
    outputs[module] = output

def bin_write(f,data):
    data = data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt,*data)
    f.write(bin)

encoder_path = os.path.realpath("./models/encoder.pth")
depth_decoder_path = os.path.realpath("./models/depth.pth")


def print_wb_output():
    if not os.path.exists('debug'):
        os.makedirs('debug')
        if not os.path.exists('debug/encoder'):
            os.makedirs('debug/encoder')
        if not os.path.exists('debug/depth_decoder'):
            os.makedirs('debug/depth_decoder')
        if not os.path.exists('debug/outputs'):
            os.makedirs('debug/outputs')
    if not os.path.exists('layers'):
        os.makedirs('layers')  

    device = torch.device("cpu") 
    encoder = networks.ResnetEncoder(18,False)
    loaded_dict_enc = torch.load(encoder_path,map_location=device)
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
     

    for n,m in encoder.named_modules():
        m.register_forward_hook(hook)

    for n,m in depth_decoder.named_modules():
        m.register_forward_hook(hook)

    encoder.eval()
    depth_decoder.eval()    
    input_image = pil.open("./images/000000.png").convert('RGB')
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    
    input_image = input_image.to(device)
    features = encoder(input_image)
    depth_outputs = depth_decoder(features)
    
    i = input_image.data.numpy()
    i = np.array(i,dtype=np.float32)
    i.tofile("debug/input.bin",format="f")


    for n,m in encoder.named_modules():
        t = '-'.join(n.split('.'))
        if m not in outputs:
            continue
        in_outputs = outputs[m]
        for i in in_outputs:
            a = []
            if  'disp' in str(i) or len(str(n)) == 0 :
                continue
            for j in i:
                if(type(j) == str):
                    continue
                else:
                    o = j.detach().numpy()
                    a.append(o)
            tempLayerArray = np.array(a,dtype=np.float32)
            t = '-'.join(n.split('.'))
            tempLayerArray.tofile("debug/encoder/"+t+".bin",format="f")
    
    for n,m in depth_decoder.named_modules():
        t = '-'.join(n.split('.'))
        if m not in outputs:
            continue
        in_outputs = outputs[m]
        for i in in_outputs:
            a = []
            if  'disp' in str(i) or len(str(n)) == 0 :
                continue
            for j in i:
                if(type(j) == str):
                    continue
                else:
                    o = j.detach().numpy()
                    a.append(o)
            tempLayerArray = np.array(a,dtype=np.float32)
            t = '-'.join(n.split('.'))
            tempLayerArray.tofile("debug/depth_decoder/"+t+".bin",format="f")


    for i,j in depth_outputs:
        a = str(i)
        b = j
        c = "output-"+a+"-"+ str(b)
        tempOutputs = depth_outputs[(a,b)].detach().numpy()
        o = np.array(tempOutputs,dtype=np.float32)
        o.tofile("debug/outputs/"+c+".bin",format="f")

           
        

if __name__ == '__main__':
    print_wb_output()