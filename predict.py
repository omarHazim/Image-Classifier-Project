import torch
from torch import nn
from torch import optim, cuda, backends
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import argparse
import json
import train
import pprint



# set up args for command line application 

def argpasr():
    paser = argparse.ArgumentParser(description='Predictions file')
    paser.add_argument('--cp', type = str, default='ImageClassifier/checkpoint1.pth', help='checkpoint path')
    paser.add_argument('--img_path', type = str, default='ImageClassifier/flowers', help='image path')
    paser.add_argument('--gpu', type = bool, default = None, help = 'GPU : True CPU: Flase')
    paser.add_argument('--arch', type = str, default = 'vgg16', help='architecture: choose btw [VGG16 or VGG13]')
    paser.add_argument('--category_names', type = str, default = None, help='Set up category names for predicted classes')
    paser.add_argument('--top_k', type = int, default = 5, help='prints top predicted classes')

    args = paser.parse_args()
    return args 


# Data processing 

def load_model(model, cp_path, device):
    if device == 'cuda':
        checkpoint = torch.load(cp_path)
    else : 
        checkpoint = torch.load(cp_path, map_location=lambda storage, loc: storage)
    print('>> model starts laoding..')
    print('-'*50)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dic'])
    print('>> model loaded successfully')
    print('-'*50)
    return model

def process_image(image_path):
    print('>> image processing started..')

    tr_img = Image.open(image_path)
    tr_img.thumbnail((100000,256))
    left = (tr_img.width - 224) / 2
    bottom = (tr_img.height - 224) / 2
    right = left + 224
    top = bottom + 224
    tr_img = tr_img.crop((left, bottom, right, top))
    np_img = np.array(tr_img) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) /std
    np_img = np_img.transpose((2,0,1))
    
    print('>> image processing finished..')
    print('-'*50)
    return np_img

def predict(processed_image, model, device, topk):
    print('>> Predicting process started ..')
    smpl_img = torch.from_numpy(processed_image).type(torch.FloatTensor)
    smpl_img.unsqueeze_(0)
    
    model.to(device)
    model.eval()
    # prediction process
    with torch.no_grad():
        smpl_img = smpl_img.to(device)
        preds = torch.exp(model.forward(smpl_img))
        top_p, top_cls = preds.topk(topk, dim=1)
        results = {'top_classes':top_cls.cpu().detach().numpy().tolist()[0], 'top_probabilities':top_p.cpu().detach().numpy().tolist()[0]}
    
    print('>> Predicting process finished ..')
    print('-'*50)
    print('>> (%s) top probabilities and classes will be printed below:' % topk)
    print('-'*50)
    return results

def get_category_lbls(model, prediction_results, jason_path):
    print('mapping class predicitons to names startd..')
    
    #process class nmae in json file
    with open(jason_path) as f:
        cat_to_name = json.load(f)
    
    # get the class labels
    idx_to_cls = {val: key for key , val in model.class_to_idx.items()}
    cls_labl = [cat_to_name[idx_to_cls[lbl]] for lbl in prediction_results['top_classes']]
    
    print('>> mapping class finished..')
    print('-'*50)
    print('>> (%s) top probabilities and class names will be printed below:' % len(cls_labl))
    print('-'*50)
    return {'class names': cls_labl, 'top_probabilities': prediction_results['top_probabilities']}

def main():
    args = argpasr()
    print('The process strated..')
    
    # set up the device
    print('-'*50)
    is_gpu = args.gpu
    # checking GPU status
    cuda_aval = torch.cuda.is_available()
    if cuda_aval:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} GPU detected..')
    if is_gpu and cuda_aval:
        device  = torch.device('cuda')
    else:
        device  = torch.device('cpu')
    print(f'>> {device} will be used in the process!')
    print('-'*50)
    
    # set up the model /classifier
    model = train.model_architecture(args.arch)
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    # Loading the model from checkpoint
    model = load_model(model, args.cp, device)
    processed_img = process_image(args.img_path)
    
    #Predict
    predictions = predict(processed_img, model, device, args.top_k)
    if args.category_names == None:
        pprint.pprint(predictions)
    else:
        predited_class_to_names = get_category_lbls(model, predictions, args.category_names)
        pprint.pprint(predited_class_to_names)
    print('-'*50)
    print('The process Completed!')


if __name__ == '__main__': main()