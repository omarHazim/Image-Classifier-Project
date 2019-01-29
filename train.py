import torch
from torch import nn
from torch import optim, cuda, backends
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import argparse
import json

# set up args for command line application 

def argpasr():
    paser = argparse.ArgumentParser(description='Taining file')
    paser.add_argument('--data_dir', type = str, default='flowers', help='dataset directory')
    paser.add_argument('--arch', type = str, default = 'vgg16', help='architecture: choose btw [VGG16 or VGG13]')
    paser.add_argument('--gpu', type = bool, default = None, help = 'GPU : True CPU: Flase')
    paser.add_argument('--hidden_units', type = int, default = 256, help = 'model hidden layer')
    paser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning rate')
    paser.add_argument('--save_dir', type = str , default = 'checkpoint1.pth', help = 'save training model')
    paser.add_argument('--epochs', type = int , default = 2, help = 'number of epochs to train the model')
    args = paser.parse_args()
    return args

# Data processing 

def process_datasets(training_data, vld_data, testing_data):
    data_transforms = transforms.Compose([transforms.RandomRotation(degrees=(-30, 30)),
                                     transforms.RandomResizedCrop(size=224),
                                     transforms.ColorJitter(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(size=224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]) 
    test_transforms = transforms.Compose([transforms.Resize(size=224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]) 
    image_datasets = {'Train':datasets.ImageFolder(training_data, transform=data_transforms),
                       'Validation':datasets.ImageFolder(vld_data, transform=valid_transforms),
                       'Test':datasets.ImageFolder(testing_data, transform=test_transforms)}
    dataloaders = {'Train': torch.utils.data.DataLoader(image_datasets['Train'], batch_size=128, shuffle=True),
                   'Validation': torch.utils.data.DataLoader(image_datasets['Validation'], batch_size=128, shuffle=True),
                   'Test': torch.utils.data.DataLoader(image_datasets['Test'], batch_size=128, shuffle=True)}
    return dataloaders, image_datasets

def data_size(image_datasets):
    datasets_size = {x: len(image_datasets[x]) for x in image_datasets.keys()}
    return datasets_size

def model_architecture(arch):
    if arch == 'vgg16' or arch == None:
        load_arch_model = models.vgg16(pretrained=True)
        print(f'>> model loaded: {load_arch_model.__class__.__name__}16' )
    elif arch == 'vgg13':
        load_arch_model = models.vgg13(pretrained=True)
        print(f'>> model loaded: {load_arch_model.__class__.__name__}13')
    else:
        print('Please choose either VGG16 of VGG13')
    return load_arch_model

def setup_classifier(model, hidden_units):
    print(f'>> Setting up a classifier for the model..')
    model.classifier[6] = nn.Sequential(nn.Linear(model.classifier[6].in_features, hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(hidden_units, 102),
                                        nn.LogSoftmax(dim=1))
    return model
        
def training_model(
    model_tr, training_set, valid_set, training_size, valid_size, criterion, optimizer, device, epochs,  
    print_every = 5):
    print('>> Training Process started!..')
    print('-'*50)
    
    steps = 0
    model_tr.to(device)
    
    # set up epochs
    if epochs == None:
        epochs = 10
    else:
        epochs
    for e in range(epochs):
        # Training process
        run_loss = 0
        for tr_img, tr_lbl in training_set:
            steps +=1
            tr_img, tr_lbl = tr_img.to(device), tr_lbl.to(device)

            optimizer.zero_grad()
            tr_logits = model_tr.forward(tr_img)
            tr_loss = criterion(tr_logits, tr_lbl)
            tr_loss.backward()
            optimizer.step()
            run_loss+= tr_loss.item() * tr_img.size(0)

            if steps % print_every == 0:
                # Validation process
                vld_loss = 0
                accu = 0.0
                model_tr.eval()
                with torch.no_grad():
                    for vld_img, vld_lbl in valid_set:
                        vld_img, vld_lbl = vld_img.to(device), vld_lbl.to(device)

                        vld_logits = model_tr.forward(vld_img)
                        vld_loss+= criterion(vld_logits, vld_lbl).item() * vld_img.size(0)
                        vld_ps = torch.exp(vld_logits)
                        top_p, top_c = vld_ps.topk(1, dim=1)
                        equal = top_c == vld_lbl.view(*top_c.shape)

                        accu += torch.mean(equal.type(torch.FloatTensor)).item() * vld_img.size(0)

                print(f'Epoch : {e+1} .. \t Training Loss: {run_loss/training_size:.3f} \t Validation Loss: {vld_loss/valid_size:.3f}'
                  f'\t Validation accuracy: {accu/valid_size:.3f}')
            run_loss = 0
            model_tr.train()
    print('>> Training Process finished!..')
    print('-'*50)
    return model_tr

def testing_model(model_tr, test_set, testing_size, criterion, device):
    print('>> Testing Process started!..')
    test_loss = 0
    accu = 0
    print('-'*50)
    
    model_tr.to(device)
    model_tr.eval()
    with torch.no_grad():
        for test_img, test_lbl in test_set:
            test_img, test_lbl = test_img.to(device), test_lbl.to(device) 
            test_logits = model_tr.forward(test_img)
            test_loss +=criterion(test_logits, test_lbl) * test_img.size(0)
            test_p = torch.exp(test_logits)
            test_prop, test_cls = test_p.topk(1, dim=1)
            test_eql = test_cls == test_lbl.view(*test_cls.shape)

            accu += torch.mean(test_eql.type(torch.FloatTensor)).item() * test_img.size(0)

        print(f'\t Testing Loss: {test_loss/testing_size:.4f} \t Testing accuracy: {accu/testing_size:.3f}')
        model_tr.train()
    print('>> Testing Process finished!..')
    print('-'*50)

def save_model(model_tr, image_datasets, optimizer, save_dir):
    if save_dir == None:
        save_dir = 'checkpoint1.pth'
        
    print(f'>> Saving {model_tr.__class__.__name__} started!..')
    print('-'*50)
    model_tr.class_to_idx = image_datasets.class_to_idx
    checkpoint = {'classifier': model_tr.classifier,
              'state_dic': model_tr.state_dict(),
              'class_to_idx': model_tr.class_to_idx,
              'optimizer' : optimizer.state_dict()}
    print(f'>> Saving finished!..')
    return torch.save(checkpoint, save_dir)

def main():
    args = argpasr()
    
    # Set up the data directories
    data_dir = 'flowers'
    training_data = data_dir + '/train'
    vld_data = data_dir + '/valid'
    testing_data = data_dir + '/test'
    
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

    # Running all operations here 
    datasets, image_datasets = process_datasets(training_data, vld_data, testing_data)
    # datasets size
    data_set_size = data_size(image_datasets)

    # set up the model /classifier
    model = model_architecture(args.arch)
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    model = setup_classifier(model, args.hidden_units)

    # set up training/validation process
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier[6].parameters(), lr = args.learning_rate)

    # Reading class labels 
    with open('cat_to_name.json') as f:
        cat_to_name = json.load(f)
            
    # Train the model
    tr_model = training_model(model, datasets['Train'], datasets['Validation'], data_set_size['Train'], data_set_size['Validation'], criterion, optimizer, device, args.epochs,  print_every = 5)

    # Testing and Saving the trained model
    testing_model(tr_model, datasets['Test'], data_set_size['Test'], criterion, device)

    save_model(tr_model, image_datasets['Train'], optimizer, args.save_dir, args.category_names)
        
    print('The process Completed!')
    
    
if __name__ == '__main__': main()
