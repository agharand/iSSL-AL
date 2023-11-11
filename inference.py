import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import splitfolders
from timeit import default_timer as timer
import time
from utils.helper_functions import (
    manual_transforms,
    knn_margin,
    query_margin_labeled,
    list_of_ints,
    plot_loss_curves
)
from utils.train_model import train
from utils.Psuedo_labels import (
    get_CIFAR10,
    get_pusedo_labels,
    get_FashionMNIST,
    get_MNIST
)
from utils.dataloader_setup import create_dataloaders
from utils.train_SSL import train
import argparse
import torchvision
import os



def SSL(dataset_name,raw_data_path,rotation_data_path,rotation_angles,device):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        SSL_train_data,SSL_test_data=get_CIFAR10()
    elif dataset_name == 'mnist':
        SSL_train_data,SSL_test_data=get_MNIST()
    elif dataset_name == 'mnist':
        SSL_train_data,SSL_test_data=get_FashionMNIST()
    else:
        print(f"{dataset_name} is not a supported dataset")

    # create the rotation data using the original dataset and the rotation angles for the SSL part
    get_pusedo_labels(raw_data_path,rotation_data_path,rotation_angles) 
    # splitting the data into train and test set and save them into dir
    splitfolders.ratio(rotation_data_path, output="rotation_data_splitted", seed=1337, ratio=(.8, .2))
    train_dir = "rotation_data_splitted/train"
    test_dir = "rotation_data_splitted/val"

    #create the ssl train and test data loader
    ssl_train_dataloader, ssl_test_dataloader, ssl_class_names = create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=manual_transforms,
                                                                               batch_size=64)
    
    SSL_model = torchvision.models.efficientnet_b0(weights=None).to(device)

    #not Freezing the base model and changing the output layer, so the model learn from the psuedo_labels
    for param in SSL_model.features.parameters():
        param.requires_grad = True

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    output_shape = 2
    # Recreate the classifier layer to match the number of classes
    SSL_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=output_shape,
                        bias=True)).to(device)

    SSL_loss_fn = nn.CrossEntropyLoss()
    SSL_optimizer = torch.optim.Adam(params=SSL_model.parameters(), lr=0.0001)
    # Start the timer
    start_time = timer()
    # Setup training and save the results
    results = train(model=SSL_model,
                        train_dataloader=ssl_train_dataloader,
                        test_dataloader=ssl_test_dataloader,
                        optimizer=SSL_optimizer,
                        loss_fn=SSL_loss_fn,
                        epochs=5,
                        device=device)

    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
    return SSL_model

def AL(device,dataset_name,SSL_model):

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        AL_train_data,AL_test_data=get_CIFAR10()
    elif dataset_name == 'mnist':
        AL_train_data,AL_test_data=get_MNIST()
    elif dataset_name == 'mnist':
        AL_train_data,AL_test_data=get_FashionMNIST()
    else:
        print(f"{dataset_name} is not a supported dataset")
    
    # Model: efficientnet_b0
    AL_model = torchvision.models.efficientnet_b0(weights='DEFAULT').to(device)

    for param in AL_model.features.parameters():
        param.requires_grad = True
  
    
    # Recreate the classifier layer to match the number of classes
    output_shape = 10
    AL_model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=10,
                        bias=True)).to(device)
    
    test_dataloader = DataLoader(dataset=AL_test_data,
                            batch_size=32,
                            num_workers=  os.cpu_count(),
                            shuffle=False) 

    # Setup loss function/eval metrics/optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=AL_model.parameters(),
                                lr=0.0001)
    
    # first AL cycle
    start = time.time()
    labeled_idxs =  np.zeros(len(AL_train_data), dtype=bool)
    sample_size = 2000
    tmp_idxs = np.arange(len(AL_train_data))
    np.random.shuffle(tmp_idxs)
    labeled_idxs[tmp_idxs[:sample_size]] = True

    train_labeled_idxs=labeled_idxs.nonzero()
    len(train_labeled_idxs[0])

    epochs=30
    train_subset = Subset(AL_train_data, train_labeled_idxs[0])
    train_dataloader = DataLoader(dataset=train_subset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    results=train(AL_model,train_dataloader,test_dataloader,optimizer,loss_fn,epochs,device)
    end=time.time()
    print(f'time for 1st cycle of AL= {end-start} sec')

    ncycles = 4
    epochs = 30
    sample_size = 2000
    cycle_model_results={}
    marign_points=[]
    for cycle in range(ncycles):
        train_time_start_model = timer()
        print(f"Cycle: {cycle+1}\n-------")
        
        l=query_margin_labeled(AL_model,sample_size,labeled_idxs,AL_train_data,device)
        marign_points.extend(l)
        sampleidx_knn=knn_margin(marign_points,labeled_idxs,sample_size,device,AL_train_data,SSL_model)
        labeled_idxs[sampleidx_knn] = True ##update
        train_subset = Subset(AL_train_data, sampleidx_knn) #subset of Al train data by using the unlabeled ids
        BATCH_SIZE = 32
        workers = os.cpu_count()
        train_dataloader = DataLoader(dataset=train_subset,
                                    batch_size=BATCH_SIZE,
                                    num_workers= workers,
                                    shuffle=True)
        train_time_start_model = timer()

        results = train(model=AL_model,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=epochs,
                            device=device)
        end_time = timer()
        print(f"[INFO] Total training time for {cycle+1} cycle of AL: {end_time-train_time_start_model:.3f} seconds for the {cycle+1} cycle")
        cycle_model_results[cycle+1]=results

    plot_loss_curves(cycle_model_results[1])

    plot_loss_curves(cycle_model_results[2])

    plot_loss_curves(cycle_model_results[3])

    plot_loss_curves(cycle_model_results[4])
     


if __name__=="__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='dataset_name', type=str, default='cifar10',
                         help="The dataset name; options: fashionmnist, mnist, cifar10")
    parser.add_argument(dest='rotation_angles', type=list_of_ints, default=['0','90'],
                         help="The rotations angles list; default is [0,90]")
    parser.add_argument(dest='rotation_data_path', type=str, default='rotations',
                         help="The rotation data path") 
    parser.add_argument(dest='raw_data_path', type=str, default='raw-data',
                         help="The raw data path")
    parser.add_argument(dest='BATCH_SIZE', type=int, default='64',
                         help="The batch size")
    
    args = parser.parse_args()
    dataset_name= args.dataset_name
    rotation_angles = args.rotation_angles
    rotation_data_path= args.rotation_data_path
    raw_data_path=args.raw_data_path
    BATCH_SIZE= args.BATCH_SIZE

    # call the SSL part
    SSL_model = SSL(dataset_name,raw_data_path,rotation_data_path,rotation_angles,device)

    # Call the AL part
    AL(device,dataset_name,SSL_model)