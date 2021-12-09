import numpy as np
from skimage.morphology import skeletonize
import sknw
import cv2
import os
from argparse import ArgumentParser
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='the path of road prediction')
    parser.add_argument('output', help='the output directory of fined-road')
    parser.add_argument(
        '--tau', type=int, default=100, help='connect points in distance tau')
    parser.add_argument(
        '--thickness', type=int, default=10, help='make lines thickening')
    parser.add_argument(
        '--angle', type=int, default=10, help='connect lines if angle matching')
    parser.add_argument(
        '--post_area_threshold', type=int, default=200, help='discard small pieces')
    args = parser.parse_args()
    return args
    
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
def patch_regular(gt,tau=100,thickness=10,angle=10):
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
def regular(img,output,tau=100,thickness=10,angle=10, post_area_threshold=200):
    name=os.path.basename(img)
    img=cv2.imread(img)
    img=remove_noise(img[:,:,0]*255,post_area_threshold)
    img_regular=patch_regular(img,tau,thickness,angle)
    img_regular=remove_noise(img_regular*255,post_area_threshold)
    cv2.imwrite(os.path.join(output,name),img_regular)

if __name__ == '__main__':
    args = parse_args()
    regular(args.img,args.output,args.tau,args.thickness,args.angle, args.post_area_threshold)
    print('success')
