"""
Training functions for iterating through one epoch.
"""


from timm.data.mixup import *
import torch
import time
from utils import *
from tqdm import tqdm
import torch.utils.data as data

def train_one_epoch(model, trainloader, optimizer, scheduler, epoch, device, mixup_fn=None, criterion=None, arcface=False):
    # iterates through one epoch of the dataset when training classification models.
    model.train()
    print('Training', flush=True)
    lrs = [] 
    training_loss = AverageMeter()
    data_loading_time = AverageMeter()
    batch_time = AverageMeter()
    
    iters = len(trainloader) # only relevant if using a CosineAnnealing scheduler.
    acc = 0
    
    start_time = time.time()
    for idx, (image, labels) in enumerate(tqdm(trainloader, position=0, leave=True)):
        # time required to load the data
        data_loading_time.update(time.time() - start_time)
        lrs.append(scheduler.get_last_lr()[0])
        
        image = image.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # cache the labels pre-mixup for accuracy calculation:
        labels_acc = labels
        
        if mixup_fn is not None:
            image, labels = mixup_fn(image, labels) #mixup
        
        optimizer.zero_grad()     
        
        with torch.cuda.amp.autocast():
            if arcface:
                loss, output = model(image, labels)     
            else:
                output = model(image)
                loss = criterion(output, labels)
            
        loss.backward()
        training_loss.update(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        
        # accuracy for each batch
        acc += accuracy(output, labels_acc, 1)
        
        # small update in learning rate -- uncomment if usine cosine
        #scheduler.step(epoch + idx / iters)
        
        #if counter % 1000 == 0:
        #    print(f'Current loss at iteration {counter}: {training_running_loss / counter}')
        
        batch_time.update(time.time() - start_time)
        start_time = time.time()
    
    # step in scheduler when using exponential
    scheduler.step()

    accuracy_train = (acc / len(trainloader.dataset)) * 100
    
    print(f'Average data loading time each batch at epoch {epoch}: {data_loading_time.avg}', flush=True)
    print(f'Average batch time at epoch {epoch}: {batch_time.avg}', flush=True)
    
    return lrs, training_loss.avg, accuracy_train, batch_time.sum


def validate_model(model, validationloader, device, arcface=False):
    # validates the classification model on the validation subset
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    print('Validating')
    validation_loss = AverageMeter()
    acc = 0
    with torch.no_grad():
        for idx, (image, labels) in enumerate(validationloader):
            image = image.to(device)
            labels = labels.to(device)
            
            if arcface:
                loss, output = model(image, labels)
            else:
                output = model(image)
                loss = criterion(output, labels)
            
            validation_loss.update(loss.item())
            acc += accuracy(output, labels, 1)
            
    accuracy_val = (acc / len(validationloader.dataset)) * 100
    
    return validation_loss.avg, accuracy_val
    
def train_triplet(model, train_dataset, optimizer, scheduler, epoch, device, criterion, model_type, two_losses, subsets_per_epoch=20):
    model.train()
    
    if model_type == 'vit':
        output_dim = 384
    elif model_type == 'cait':
        output_dim = 288
    else:
        output_dim = 576
    # divides dataset into smaller cached sets of randomly sampled indices
    train_dataset.new_epoch()
    training_loss = AverageMeter()
    data_loading_time = AverageMeter()
    batch_time = AverageMeter()
    
    lrs = []
    
    optimizer.zero_grad()

    # subset_per_epoch to None if iterating through the entire dataset for one epoch is desired
    # only iterating through a lower number of subsets as to allow convenient and consistend saving/loading
    if subsets_per_epoch == None or subsets_per_epoch > train_dataset.nCacheSubset:
        subsets_per_epoch = train_dataset.nCacheSubset
        initial_subset = 0
    else: # starting from a random subset, shouldnt really be necessary
        initial_subset = np.random.randint(train_dataset.nCacheSubset-subsets_per_epoch)
        train_dataset.current_subset = initial_subset
    print(f'Training for {subsets_per_epoch} subsets out of {train_dataset.nCacheSubset} in total')
    
    

    while train_dataset.current_subset < (initial_subset + subsets_per_epoch):
        train_dataset.update_subcache(output_dim, model)
        model.train()
        # create data loader
        # given the random sampling of .new_epoch() shuffling the dataloader shouldnt be necessary.
        trainDataloader = data.DataLoader(train_dataset, batch_size=8, shuffle=False, # prøv ut true 
                                          num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_tuples)
        
        lrs.append(scheduler.get_last_lr()[0])
        # visualize a triplet
        start_time=time.time()

        for idx, (input, target) in enumerate(tqdm(trainDataloader, position=0, leave=True, desc=f'epoch subset {train_dataset.current_subset} of {initial_subset+subsets_per_epoch}')):
            data_loading_time.update(time.time() - start_time)
            nq = len(input) # number of training tuples
            ni = len(input[0]) # number of images per tuple

            for q in range(nq):
                output_cls = torch.zeros(output_dim, ni).cuda()
                if two_losses:
                    output_dist = torch.zeros(output_dim, ni).cuda()
                for imi in range(ni):
                    # compute output vector for image imi
                    if two_losses:
                        output_cls[:, imi], output_dist[:, imi] = model(input[q][imi].unsqueeze(0).cuda()) #.squeeze()
                    else:
                        output_cls[:, imi] = model(input[q][imi].unsqueeze(0).cuda()).squeeze()
                        
                # reducing memory consumption:
                # compute loss for this query tuple only
                # then, do backward pass for one tuple only
                # each backward pass gradients will be accumulated
                # the optimization step is performed for the full batch later
                
                if two_losses:
                    # keep only one positive
                    output_cls = output_cls.transpose(0,1)
                    output_cls = output_cls[torch.arange(output_cls.size(0))!=2]
                    output_dist = output_dist.transpose(0,1)
                    output_dist = output_dist[torch.arange(output_dist.size(0))!=1]

                    loss = criterion(output_cls.transpose(0,1), output_dist.transpose(0,1), target[q].cuda())
                else:
                    loss = criterion(output_cls, target[q].cuda())

                training_loss.update(loss.item())
                loss.backward()
            
            optimizer.step()

            optimizer.zero_grad()
            
            batch_time.update(time.time() - start_time)
            start_time = time.time()

        if train_dataset.current_subset % 5 == 0:
            #print('------------')
            print('\n',f'loss: {training_loss.avg}')
        #scheduler.step(epoch + train_dataset.current_subset / subsets_per_epoch)

    # finalizing epoch
    scheduler.step()

    print(f'average data loading time: {data_loading_time.avg}')
    print(f'average batch time: {batch_time.avg}')
    
    return training_loss.avg, lrs, batch_time.sum

