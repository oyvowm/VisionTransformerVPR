"""
Evaluation script based on https://github.com/MubarizZaffar/VPR-Bench/blob/main/performance_comparison.py

Modified to work for the trained pytorch models

Running e.g. 'python evaluate.py -d SPEDTEST -m all -pre True' will evaluate all models contained within the models folder on the SPEDTEST dataset (assuming that no precomputed data is available), while also including the results of the other VPR-techniques. 
"""



import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import csv
import time
import math
from datetime import datetime
import os
import torch
import torch.nn as nn
from model import *
from pit import PoolingTransformer, DistilledPoolingTransformer
from cait import cait_models, cait_models_twoQ
from datasets import EvaluationDataset
import torch.utils.data as data
import torchvision
from torchvision import transforms
from tqdm import tqdm
import argparse

def csv_summary(model, dataset, auc, encoding_time, matching_time):
    with open('results/csv_summary/'+dataset+'.csv', 'a') as csvfile:
        csv_writer = csv.writer(csvfile)
        row = [str(dataset),str(model),str(auc),str(encoding_time),str(matching_time)]
        csv_writer.writerow(row)


def compute_precision_recall(matches,scores_all):
    precision, recall, _ = precision_recall_curve(matches, scores_all)   
    return precision, recall

def compute_roc(matches,scores_all):
    fpr,tpr, _=roc_curve(matches,scores_all)
    return fpr, tpr

def compute_auc_ROC(matches,scores_all):
    if (np.sum(matches)==len(matches)): #All images are true-positives in the dataset
        print('Only single class for ROC! Computation not possible.')
        return -1
    else:
        return roc_auc_score(matches,scores_all) #This throws an error when all images are true-positives because then it is a single class only.

def compute_auc_PR(prec,recall): #Area-under-the-Precision-Recall-Curve
    return auc(recall, prec)

def compute_accuracy(matches):
    accuracy=float(np.sum(matches))/float(len(matches))
    return accuracy

def compute_RecallRateAtN_forRange(all_retrievedindices_scores_allqueries, ground_truth_info):
    sampleNpoints=range(1,20) #Can be changed to range(1,0.1*len(all_retrievedindices_scores_allqueries[0])) for maximum N equal to 10% of the total reference images
    recallrate_values=np.zeros(len(sampleNpoints))
    itr=0
    for N in sampleNpoints:      
        recallrate_values[itr]=compute_RecallRateAtN(N, all_retrievedindices_scores_allqueries, ground_truth_info)
        itr=itr+1
    
    #print(recallrate_values)
    return recallrate_values, sampleNpoints


def compute_RecallRateAtN(N, all_retrievedindices_scores_allqueries, ground_truth_info):
    matches=[]
    total_queries=len(all_retrievedindices_scores_allqueries)
    match_found=0
    
    for query in range(total_queries):
        top_N_retrieved_ind=np.argpartition(all_retrievedindices_scores_allqueries[query], -1*N)[-1*N:]
        for retr in top_N_retrieved_ind:        
            if (retr in ground_truth_info[query][1]):
                match_found=1
                break

        if (match_found==1):
            matches.append(1)
            match_found=0
        else:
            matches.append(0)            
            match_found=0
     
    recallrate_at_N=float(np.sum(matches))/float(total_queries)
    
    return recallrate_at_N

def compute_matches(retrieved_all, ground_truth_info):
    matches=[]
    itr=0
    for retr in retrieved_all:
        if (retr in ground_truth_info[itr][1]):
            matches.append(1)
        else:
            matches.append(0)
        itr=itr+1        
    return matches

def draw_RecallRateAtKCurves(recallrate_at_K_dict,sampleNpoints,models,dataset):
    plt.figure()
    for model in models:
    
        plt.plot(sampleNpoints, recallrate_at_K_dict[model], label=model)
        plt.legend(loc='lower right', fontsize='large')          

        plt.xlabel('N',fontsize='x-large')
        plt.ylabel('RecallRate',fontsize='x-large')
    
    plt.title(dataset)
    plt.grid()     
    plt.savefig('results/RecallRateCurves/'+dataset+'-RecallRateCurves'+'.png') 
#    plt.show()     

def draw_ROC_Curves(fpr_dict,tpr_dict,models,dataset):
    plt.figure()
    for model in models:
    
        plt.plot(fpr_dict[model], tpr_dict[model], label=model)
        plt.legend(loc='lower right', fontsize='large')          

        plt.xlabel('FPR',fontsize='x-large')
        plt.ylabel('TPR',fontsize='x-large')
    
    plt.title(dataset)
    plt.grid()     
    plt.savefig('results/ROCCurves/'+dataset+'-ROCcurves'+'.png') 
#    plt.show()   

def draw_PR_Curves(prec_dict,recall_dict,dataset, models):   
    plt.figure()

    for model in models:
        
        plt.plot(recall_dict[model], prec_dict[model], label=(model))
        plt.legend(loc='lower right', fontsize='large')          

        plt.xlabel('Recall',fontsize='x-large')
        plt.ylabel('Precision',fontsize='x-large')
    
    plt.xlim(0,1)
    plt.ylim(-0.05,1.05)
    plt.title(dataset)    
    plt.grid()     
    plt.savefig('results/PRCurves/'+dataset+'_PRcurves'+'.png') 
#    plt.show()    


def load_model(model_name):
    """
    Loads the models to be evaluated
    To evaluate models with different hyperparameters the model config dicts can be changed accordingly.
    Other model naming conventions can be used by updating the if statements.
    """
    vit_config = {
            "img_size": (224,224),
            "in_chans": 3,
            "patch_size": 16,
            "embed_dim": 384,
            "triplet": False,
            "depth": 12,
            "num_heads": 6,
            "qkv_bias": True,
            "hidden_mult": 4,
            "num_classes": 900,
            "embed_fn": 'vit' # 'convolution' for a convolutional backbone
    }
    pit_config = {
        "img_size": (224,224),
        "patch_size": 16,
        "stride": 8,
        "num_classes": 900,
        "base_dims": [48, 48, 48],
        "depth": [2, 6, 4],
        "heads": [3, 6, 12],
        "mlp_ratio": 4,
        "triplet": False,
        "embed_fn": "vit", # 'convolution' for a convolutional backbone
        "residual_scaling": 1.,
}
    
    cait_config = {
        "img_size": (224,224),
        "embed_dim": 288,
        "depth": 24,
        "num_heads": 6,
        "qkv_bias": True,
        "num_classes": 900,
        "init_scale": 1e-5,
        "depth_token_only": 2,
        "triplet": False,
}


    t1 = time.time()
    if model_name[-7:] == 'pit.pth':
        if model_name[-15:-8] == 'triplet':
            pit_config["triplet"] = True
            model = DistilledPoolingTransformer(**pit_config)
        elif model_name[-23:-8] == 'tripletNewPatch': # if using a convolutional backbone
            pit_config["triplet"] = True
            pit_config["embed_fn"] = 'convolution'
            pit_config["residual_scaling"] = 2.
            model = DistilledPoolingTransformer(**pit_config)
        else:
            model = DistilledPoolingTransformer(**pit_config)
            
    elif model_name[-7:] == 'vit.pth':
        if model_name[-15:-8] == 'arcface':
            model = ArcFaceDeitEval(**vit_config)
        elif model_name[-15:-8] == 'triplet':
            vit_config["triplet"] = True
            model = DistilledVisionTransformer(**vit_config)      
        else:
            vit_config.pop('embed_fn')
            model = DistilledVisionTransformer(**vit_config)
            
    elif model_name[-8:] == 'cait.pth':
        if model_name[:7] == 'triplet':
            cait_config["triplet"] = True
            if model_name[8:13] == 'twoQs':
                model = cait_models_twoQ(**cait_config)
            else:
                model = cait_models(**cait_config)
        else:
            model = cait_models(**cait_config)
            
    save_path = os.path.join(os.getcwd(), 'models', model_name) 

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    checkpoint = torch.load(save_path, map_location = device)
    
    # load model
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    # removes the classification layer
    try:
        model.head = nn.Identity()
    except:
        print('Failed changing model head to Identity')
    if model_name[-8:] != 'cait.pth':
        try:
            model.head_dist = nn.Identity()
        except:
            print('Failed changing model dist head to Identity')
            
    model.to(device)
    model.eval()
    t2 = time.time()
    print(f'loading model ({model_name}) in time: {t2-t1}')

    return model



def compute_reference_features(model, dataset_path, transform):
    """
    Function that extract feature vectors for all images in the reference set
    """
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    # create dataloader
    reference_path = os.path.join(dataset_path, 'ref')
    
    reference_dataset = EvaluationDataset(reference_path, transform)
                               
    reference_dataloader = data.DataLoader(reference_dataset, batch_size=1, num_workers=0, shuffle=None)
                               
    ref_image_features = []
    with torch.no_grad():                           
        for image in tqdm(reference_dataloader):
            image = image.to(device)
            feature = model(image)
            feature = feature.cpu().detach()
            feature = feature / np.linalg.norm(feature)
            ref_image_features.append(feature)
    return ref_image_features
   
    
def compute_query_desc(model, query_image):
    """
    Function that extracts the feature vector of a single query image
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    with torch.no_grad():
        query_image = query_image.to(device)
        query_feature = model(query_image)
        query_feature = query_feature.cpu().detach()
        query_feature = query_feature / np.linalg.norm(query_feature)
    
    return query_feature

def perform_VPR(query, reference_features):
    """
    Calculates the matching scores of a query descriptor and all referece descriptors
    """
    matching_scores = []
    
    for feature_vector in reference_features:
        score = np.dot(query, feature_vector.T)
        matching_scores.append(score)
        
    return np.amax(matching_scores), np.argmax(matching_scores), np.asarray(matching_scores).reshape(len(reference_features)) 
        


def place_match(model,query_image,robot_map_features):
    """
    Ties together and times the different components of the retrieval process.
    """
    t1=time.time()
    query_desc=compute_query_desc(model,query_image)
    t2=time.time()
    matching_score,matched_vertex, confusion_vector=perform_VPR(query_desc,robot_map_features)  
    t3=time.time()
    
    return 1,matched_vertex,matching_score,(t2-t1),((t3-t2)/len(robot_map_features)), confusion_vector

def performance_comparison(dataset_name, retrieved_all, scores_all, encoding_time_all, matching_time_all, all_retrievedindices_scores_allqueries_dict, model): 
    """
    Extracts and calculates the performance on a number of metrics using the matching data.
    """
    
    ground_truth_info=np.load(os.path.join(os.getcwd(), 'datasets', dataset_name, 'ground_truth_new.npy'),allow_pickle=True) # A 2-dimensional array representing the range of reference image matches correponding to a query image   

                        
    matches=compute_matches(retrieved_all, ground_truth_info)
    prec,recall=compute_precision_recall(matches,scores_all)
    auc_pr=compute_auc_PR(prec,recall)
    recallrate_at_K_range, sampleNpoints=compute_RecallRateAtN_forRange(all_retrievedindices_scores_allqueries_dict, ground_truth_info) #K range is 1 to 20
    accuracy=compute_accuracy(matches)
    fpr,tpr=compute_roc(matches,scores_all)
    auc_roc=compute_auc_ROC(matches,scores_all)
        
    # write to csv file
    csv_summary(model, dataset_name, auc_pr, encoding_time_all, matching_time_all)

    
    print(f'accuracy of {model} is: {accuracy:.2f}')
    print(f'AUC PR {model}: {auc_pr:.2f}')
    print(f'AUC ROC {model}: {auc_roc:.2f}')
    print(f'Encoding Time {model}: {encoding_time_all:.3f}')
    print(f'Matching Time {model}: {matching_time_all}', '\n')
    

    return prec, recall, recallrate_at_K_range, sampleNpoints, fpr, tpr, auc_roc

    
def main():

    models = {}
    ref_image_descriptors = {}
    
    if (args['model_names'][0]) == 'all':
        print('>>> No models specified, using all models in directory')
        model_names = os.listdir(os.path.join(os.getcwd(), 'models'))
    else:
        model_names = args['model_names']
    
    if args['use_precomputed'] == 'True' or args['use_precomputed'] == 'true':
        pre_computed_models = ['AlexNet_VPR', 'AMOSNet', 'ap-gem-r101', 'CALC', 'CoHOG', 'denseVLAD', 'HOG', 'HybridNet', 'NETVLAD', 'RegionVLAD']
        #pre_computed_models = ['NETVLAD'] # uncomment if only plotting NetVLAD next to the trained models.
        model_names.extend(pre_computed_models)
    print(model_names)

    matching_indices = {}
    matching_scores = {}
    query_indices = {}
    encoding_times = {}
    matching_times = {}
    confusion_matrices = {}
    descriptor_shape_dict={}
    
    precision = {}
    recall = {}
    recallrate_at_K_dict={}
    sampleNpoints=[]
    fpr = {}
    tpr = {}
    auc_roc = {}
    
    # initiate csv file
    with open('results/csv_summary/'+dataset_name+'.csv', 'w') as csvfile:
        fieldnames = ['dataset', 'model name', 'auc', 'encoding time', 'matching time']
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

    for model in model_names:
    
        try:
            precomputed_data=np.load(os.getcwd()+'/'+'precomputed'+'/' + dataset_name + '_precomputed_data_' + model + '.npy',allow_pickle=True, encoding='latin1') # latin1 encoding to allow loading the data saved in Python 2 numpy files.
            print(f'using precomputed features for model {model}')
            query_indices[model] = precomputed_data[0]        
            matching_indices[model] = precomputed_data[1]
            matching_scores[model] = precomputed_data[2]
            confusion_matrices[model] = precomputed_data[3]
            encoding_times[model] = precomputed_data[4]  
            matching_times[model] = precomputed_data[5] 

        except:
            print(f'Computing new feature extractions for model {model}')
            
            models[model] = load_model(model)
            transform_image = transforms.Compose([
                                  transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #imagenet
                                             ])

            # create query dataloader
            query_path = os.path.join(dataset_path, 'query')
            query_dataset = EvaluationDataset(query_path, transform_image)                              
            query_dataloader = data.DataLoader(query_dataset, batch_size=1, num_workers=0, shuffle=None)
            print(len(query_dataloader))
            
            ref_image_descriptors[model] = compute_reference_features(models[model], dataset_path, transform_image)
            matching_indices_list = []
            matching_scores_list = []
            query_indices_list = []
            encoding_time = []
            matching_time = []        
            confusion_matrix=[]

            for idx, query_image in enumerate(tqdm(query_dataloader)):
                matched, matching_index, score, t_e, t_m, confusion_vector = place_match(models[model], query_image, ref_image_descriptors[model])  #Matches a given query image with all reference images.
                query_indices_list.append(idx)
                matching_indices_list.append(matching_index)
                matching_scores_list.append(score)
                confusion_matrix.append(confusion_vector)
                encoding_time.append(t_e)  #Feature Encoding time per query image
                matching_time.append(t_m)  #Descriptor Matching time for 2 image descriptors    
                descriptor_shape=str(ref_image_descriptors[model][0].shape)+' '+str(ref_image_descriptors[model][0].dtype)

            encoding_time_avg = sum(encoding_time) / len(encoding_time)
            matching_time_avg = sum(matching_time) / len(matching_time)
            matching_indices[model] = matching_indices_list
            matching_scores[model] = matching_scores_list
            query_indices[model] = query_indices_list
            encoding_times[model] = encoding_time_avg
            matching_times[model] = matching_time_avg
            confusion_matrices[model] = confusion_matrix
            descriptor_shape_dict[model]=descriptor_shape

            precomputed_data=np.asarray([np.asarray(query_indices_list), np.asarray(matching_indices_list), np.asarray(matching_scores_list), np.asarray(confusion_matrix), encoding_time_avg, matching_time_avg], dtype="object")

            # save the computed data
            np.save(os.getcwd()+'/'+'precomputed'+'/'+ dataset_name +'_precomputed_data_' + model + '.npy',precomputed_data)

            

        # run evaluation
        precision[model], recall[model],  recallrate_at_K_dict[model], sampleNpoints, fpr[model], tpr[model], auc_roc[model] = performance_comparison(dataset_name, matching_indices[model], matching_scores[model], encoding_times[model], matching_times[model], confusion_matrices[model], model)
        

        
 
        
    draw_PR_Curves(precision,recall,dataset_name, model_names) 
    draw_RecallRateAtKCurves(recallrate_at_K_dict,sampleNpoints,model_names,dataset_name)
    draw_ROC_Curves(fpr,tpr,model_names,dataset_name) 
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataset', default='SPEDTEST', required=True, help='Name of the dataset to use', type=str)
    parser.add_argument('-m', '--model_names', nargs= '+', help='Name of the model/s to load -> "all" uses all models in directory.', type=str)
    parser.add_argument('-pre', '--use_precomputed', default='False', help='Whether or not to include pre-computed matches from VPR-Bench.', type=str)

    args = vars(parser.parse_args())
    
    dataset_name = args["dataset"]
    dataset_path = os.path.join(os.getcwd(), 'datasets', dataset_name)
    
    main()
