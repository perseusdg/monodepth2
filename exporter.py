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
import matplotlib.pyplot as plt

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
    f = None
    f2 = None
    open_conv_twice = False
    bias_shape = 0
    if not os.path.exists('tkDNN_bin'):
        os.makedirs('tkDNN_bin')  
    if not os.path.exists('tkDNN_bin/debug'):
        os.makedirs('tkDNN_bin/debug')
        if not os.path.exists('tkDNN_bin/debug/encoder'):
            os.makedirs('tkDNN_bin/debug/encoder')
        if not os.path.exists('tkDNN_bin/debug/depth_decoder'):
            os.makedirs('tkDNN_bin/debug/depth_decoder')
        if not os.path.exists('tkDNN_bin/debug/outputs'):
            os.makedirs('tkDNN_bin/debug/outputs')
    if not os.path.exists('tkDNN_bin/layers'):
        os.makedirs('tkDNN_bin/layers')
        if not os.path.exists('tkDNN_bin/layers/encoder'):
            os.makedirs('tkDNN_bin/layers/encoder')
        if not os.path.exists('tkDNN_bin/layers/depth_decoder'):
            os.makedirs('tkDNN_bin/layers/depth_decoder')


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
    input_image = pil.open("dog.jpg").convert('RGB')
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    
    input_image = input_image.to(device)
    features = encoder(input_image)
    depth_outputs = depth_decoder(features)
    
    i = input_image.data.numpy()
    i = np.array(i,dtype=np.float32)
    print('input', np.shape(i))
    i.tofile("tkDNN_bin/debug/input.bin",format="f")


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
            print(n, np.shape(tempLayerArray))
            tempLayerArray.tofile("tkDNN_bin/debug/encoder/"+t+".bin",format="f")
    
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
            print(n, np.shape(tempLayerArray), t)
            tempLayerArray.tofile("tkDNN_bin/debug/depth_decoder/"+t+".bin",format="f")

    for i,j in depth_outputs:
        a = str(i)
        b = j
        c = "output-"+a+"-"+ str(b)
        tempOutputs = depth_outputs[(a,b)].detach().numpy()
        o = np.array(tempOutputs,dtype=np.float32)
        # plt.imshow(np.squeeze(np.squeeze(o, axis=0), axis=0))
        # plt.show()
        o.tofile("tkDNN_bin/debug/outputs/"+c+".bin",format="f")


      
    for n,m in encoder.named_modules():
        t = '-'.join(n.split('.'))

        if not(' of Conv2d' in str(m.type) or ' of BatchNorm2d' in str(m.type) or ' of Linear' in str(m.type)):
            continue

        if ' of Conv2d' in str(m.type) or ' of Linear' in str(m.type) or ' of Sigmoid' in str(m.type):
            file_name = "tkDNN_bin/layers/encoder/" + t + ".bin"
            print("open file: ", file_name)
            f = open(file_name, mode='wb')
        
        if 'of BatchNorm2d' in str(m.type):
            b = m._parameters['bias'].cpu().data.numpy()
            b = np.array(b, dtype=np.float32)
            s = m._parameters['weight'].cpu().data.numpy()
            s = np.array(s, dtype=np.float32)
            rm = m.running_mean.cpu().data.numpy()
            rm = np.array(rm, dtype=np.float32)
            rv = m.running_var.cpu().data.numpy()
            rv = np.array(rv, dtype=np.float32)
            bin_write(f, b)
            bin_write(f, s)
            bin_write(f, rm)
            bin_write(f, rv)

            print("    b shape:", np.shape(b))
            print("    s shape:", np.shape(s))
            print("    rm shape:", np.shape(rm))
            print("    rv shape:", np.shape(rv))
        else:
            w = np.array([])
            b = np.array([])
            if 'weight' in m._parameters and m._parameters['weight'] is not None:
                w = m._parameters['weight'].cpu().data.numpy()
                w = np.array(w, dtype=np.float32)
                print("    weights shape:", np.shape(w))

            if 'bias' in m._parameters and m._parameters['bias'] is not None:
                b = m._parameters['bias'].cpu().data.numpy()
                b = np.array(b, dtype=np.float32)
                print("    bias shape:", np.shape(b))
                
            bin_write(f, w)
            bias_shape = w.shape[0]
            if b.size > 0:
                bin_write(f, b)

        if ' of BatchNorm2d' in str(m.type) or ' of Linear' in str(m.type):
            f.close()
            print("close file")
            f = None

    f = None
    bias_shape = 0
    for n,m in depth_decoder.named_modules():
        t = '-'.join(n.split('.'))
        if not( ' of Conv2d' in str(m.type) or ' of BatchNorm2d' in str(m.type) or ' of ELU' in str(m.type) or 'of Linear' in str(m.type) or 'of Sigmoid' in str(m.type)):
            continue
        if (' of Conv2d' in str(m.type) or ' of ELU' in str(m.type) or 'of Sigmoid' in str(m.type)):
            if f is not None:
                if bias_shape != 0:
                    bin_write(f, np.zeros(bias_shape))
                    bias_shape = 0
                f.close()
                print("close file")
                f = None

            file_name = "tkDNN_bin/layers/depth_decoder/" + t + ".bin"
            print("open file: ", file_name)
            f = open(file_name, mode='wb')
        
        w = np.array([])
        b = np.array([])
        if 'weight' in m._parameters and m._parameters['weight'] is not None:
            w = m._parameters['weight'].cpu().data.numpy()
            w = np.array(w, dtype=np.float32)
            print("    weights shape:", np.shape(w))
            

        if 'bias' in m._parameters and m._parameters['bias'] is not None:
            b = m._parameters['bias'].cpu().data.numpy()
            b = np.array(b, dtype=np.float32)
            print("    bias shape:", np.shape(b))
        
        bin_write(f, w)
        bias_shape = w.shape[0]
        if b.size > 0:
            bin_write(f, b)
            bias_shape = 0
        elif b.size == 0:
            bin_write(f, np.zeros(bias_shape))
            bias_shape = 0
        f.close()
        print("close file")
        f = None


if __name__ == '__main__':
    print_wb_output()