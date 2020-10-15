import argparse
import json
import os
import numpy as np
import utils
from task1 import *

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task2.jpg",
        help="path to the image")
    parser.add_argument(
        "--template-path",
        type=str,
        default="./data/proj1-task2-template.jpg",
        help="path to the template"
    )
    parser.add_argument(
        "--result-saving-path",
        dest="rs_path",
        type=str,
        default="./results/task2.json",
        help="path to file which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args

def sss(img_patch,templ):
    patch_avg=0
    for i in img_patch:
        for j in i:
            patch_avg+=j
    patch_avg=patch_avg/(len(templ)*len(templ[0]))
    for i in range(len(templ)):
        for j in range(len(templ[0])):
            img_patch[i][j]-=patch_avg
    return img_patch

def norm_xcorr2d(patch, template):
    """Computes the NCC value between a image patch and a template.

    The image patch and the template are of the same size. The formula used to compute the NCC value is:
    sum_{i,j}(x_{i,j} - x^{m}_{i,j})(y_{i,j} - y^{m}_{i,j}) / (sum_{i,j}(x_{i,j} - x^{m}_{i,j}) ** 2 * sum_{i,j}(y_{i,j} - y^{m}_{i,j})) ** 0.5
    This equation is the one shown in Prof. Yuan's ppt.

    Args:
        patch: nested list (int), image patch.
        template: nested list (int), template.

    Returns:
        value (float): the NCC value between a image patch and a template.
    """
    #For calculating average of template values
    template_avg=0
    for i in template:
        for j in i:
            template_avg+=j
    template_avg=template_avg/(len(template)*len(template[0]))
        
    #For calculating template-template_avg and square of those values
    template_sq=[]
    for i in range(len(template)):
        for j in range(len(template[0])):
            template[i][j]-=template_avg
            template_sq.append(template[i][j]*template[i][j])
            
            
    #For caluculating sum of template-squares
    template_sqsum=0
    for i in template_sq:
        template_sqsum+=i
    
    #Multiplying template with image patch
    b=[]
    c=[]
    e=[]
    #for i in patch:    
    e.append(sss(patch,template))
    for i in e:
        b.append(utils.elementwise_mul(template,i))
        #patch square sum
        patch_square_sum=0
        for k in i:
            for j in k:
                patch_square_sum=patch_square_sum+(j*j)
        e=[]
        patch_sum=0
        for r in b:
            for k in r:
                for j in k:
                    patch_sum+=j
        c.append(patch_sum)
        b=[]
        ncc=c/(np.sqrt(patch_square_sum*template_sqsum))

    return ncc


def match(img, template):
    """Locates the template, i.e., a image patch, in a large image using template matching techniques, i.e., NCC.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        x (int): row that the character appears (starts from 0).
        y (int): column that the character appears (starts from 0).
        max_value (float): maximum NCC value.
    """
    m=len(template)
    n=len(template[0])
    x=len(img)
    y=len(img[0])
    x=x-m+1
    y=y-n+1
    int_img=[]
    for i in range(x):
        for j in range(y):
            a=[te[j:j+n] for te in img[i:i+m]]
            int_img.append(a)
    
    large=-2
    ax=0
    ay=0
    for i in range(len(int_img)):
        ay+=1
        if ((i+1)%x==0):
            ax+=1
            ay=0
        
        val=(norm_xcorr2d(int_img[i],template))
        if val>large:
            large=val
            size=i
    ay=size%y
    ax=int(size/y)
    large=large[0]
    return ax,ay,large

def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    # template = utils.crop(img, xmin=10, xmax=30, ymin=10, ymax=30)
    # template = np.asarray(template, dtype=np.uint8)
    # cv2.imwrite("./data/proj1-task2-template.jpg", template)
    template = read_image(args.template_path)

    x, y, max_value = match(img, template)
    # The correct results are: x: 17, y: 129, max_value: 0.994
    with open(args.rs_path, "w") as file:
        json.dump({"x": x, "y": y, "value": max_value}, file)


if __name__ == "__main__":
    main()
