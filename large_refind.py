from pathlib import Path
import math
import numpy as np
from skimage.morphology import skeletonize
import sknw
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import numpy as np
from numpy import ma
import skimage
import os
import cv2
import numpy as np
import torch
import rasterio as rio
import torchvision.transforms as T
from model import DinkNet34_more_dilate,DinkNet34,DinkNet50
from rsnet.dataset import RasterSampleDataset
from tqdm import tqdm
from collections import OrderedDict
def main(fname,out_dir):
    win_size = 256
    pad_size = 64

    dataset = RasterSampleDataset(fname,
                                  win_size=win_size,
                                  step_size=win_size,
                                  pad_size=pad_size)
    out_file = os.path.join(out_dir,f'{Path(fname).stem}_regular.tif')
    
    kwds = dataset.meta
    kwds.update(dict(count=1, compress='lzw', dtype='uint8'))
    out_raster = rio.open(out_file, 'w', **kwds)
    for img, xoff, yoff in tqdm(dataset):
        output=regular(img)
        pred_label = output[pad_size:-pad_size, pad_size:-pad_size]
        write_window = rio.windows.Window(xoff, yoff, win_size, win_size)
        out_raster.write(pred_label, indexes=1, window=write_window)
    out_raster.close()

def distance(x,y):
    return np.sqrt(np.sum(np.square([x[0]-y[0],x[1]-y[1]])))
def remove_noise(img, post_area_threshold):
    contours, hierarch = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        length = cv2.arcLength(contours[i], True)
        if area <= post_area_threshold or length <= post_area_threshold:
            cnt = contours[i]
            cv2.drawContours(img, [cnt], 0, 0, -1)
    img[img!=0]=1
    return img
 
def patch_regular(gt,tau,thickness,angle):
    ske = skeletonize(gt).astype(np.uint16)
    graph = sknw.build_sknw(ske)
    points=[]
    nodes=set()
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']
        if len(ps)<11:continue
        p1=[float(ps[0,0]),float(ps[0,1])]
        p1_pre=[float(ps[10,0]),float(ps[10,1])]
        k_p1=90 if p1[0]==p1_pre[0] else np.arctan(-float((p1[1]-p1_pre[1])/(p1[0]-p1_pre[0])))*57.29577
        p1.append(k_p1)
        p2=[float(ps[-1,0]),float(ps[-1,1])]
        p2_pre=[float(ps[-10,0]),float(ps[-10,1])]
        k_p2=90 if p2[0]==p2_pre[0] else np.arctan(-float((p2[1]-p2_pre[1])/(p2[0]-p2_pre[0])))*57.29577
        p2.append(k_p2)
        nodes.add(str(p1))
        nodes.add(str(p2))
        points.append({str(p1),str(p2)})
        for i in range(0,len(ps)-1):
            cv2.line(gt,(int(ps[i,1]),int(ps[i,0])), (int(ps[i+1,1]),int(ps[i+1,0])), 1,thickness=thickness)
    ps=[eval(i) for i in list(nodes)]
    for num in range(len(ps)):
        for other in range(len(ps)):
            if other!=num and {str(ps[num]),str(ps[other])} not in points:
                dis= distance(ps[num][:2],ps[other][:2])
                if dis<tau and abs(ps[num][2]-ps[other][2])<angle:
                    cv2.line(gt,(int(ps[num][1]),int(ps[num][0])), (int(ps[other][1]),int(ps[other][0])), 1,thickness=thickness)
    return gt
def regular(img,tau=100,thickness=10,angle=10, post_area_threshold=200):
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (remove_obj_holes, remove_obj_holes))
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img=remove_noise(img[:,:,0]*255,post_area_threshold)
    img_regular=patch_regular(img,tau,thickness,angle)
    img_regular=remove_noise(img_regular*255,post_area_threshold)
    return img_regular
def distance(x,y):
    return np.sqrt(np.sum(np.square([x[0]-y[0],x[1]-y[1]])))

if __name__ == '__main__':
    img_path="testdata_pred/wugang_pred_gonglu.tif"
    out_dir="testdata_pred"
    main(img_path,out_dir)

