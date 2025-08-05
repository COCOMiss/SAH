import logging
from dataset import *
from model_devide import *
from metrics import batch_performance,MRR_metric
from utils import *
from adaptive_par import AdaptiveParameter



def train(args,model,dataset,criterion,dataloader,optimizer,specific_parameters_set,task_groups,divide):
    
    
    method = AdaptiveParameter(
                device=args.device,
                n_tasks=sum(args.divide_group.values())
            )

    train_loss=0.0
    model.train()
    dataloaders = list(dataloader.values())  
    iteration_num=len(dataloaders[0])

    devide_loss=torch.zeros(sum(args.divide_group.values()), device=args.device)
    for i in range(iteration_num):
        
        optimizer.zero_grad()
        batchs=[]
        for group_num in range(len(args.divide_group)):
            batchs.append(next(iter(dataloaders[group_num])))
    
        
        
        if  divide:
            
            group_loss =train_woMeta(args,model,dataset,criterion,batchs)
            if i<=args.accu_loss:
                devide_loss=group_loss+devide_loss
            if i==args.accu_loss:

                # 获取新的任务特定参数
                original_len=len(specific_parameters_set)
                task_groups,specific_parameters_set=method.task_specific_params(losses=devide_loss,
                        model=model,
                        specific_parameters_set=specific_parameters_set)

                
                updated_len=len(specific_parameters_set)
                if updated_len-original_len > 0:
                    if updated_len-original_len > 0:
                        model.add_task_specific_params(task_groups)
                    logging.info('Create new optimizer')     
                    optimizer =torch.optim.Adam(params=list(model.parameters()),
                           lr=args.lr)
                group_loss=devide_loss
                    
                    
                
            if i>=args.accu_loss:
                loss = torch.sum(group_loss)
                loss.backward()
                for task_idx, task_params in model.task_specific_parameters.items():
                    for param_idx, param in task_params.items():
                        if param.grad is not None:
                            logging.info("update MSAHG {} task specific param {} grad {}".format(task_idx,param_idx,param.grad))
                        else:
                            logging.info("MSAHG {} task specific param {} grad is None".format(task_idx,param_idx))
                optimizer.step()
                optimizer.zero_grad()

              
        else:
            
            group_loss =train_woMeta(args,model,dataset,criterion,batchs)
            
            loss = torch.sum(group_loss)
            loss.backward()
            for task_idx, task_params in model.task_specific_parameters.items():
                for param_idx, param in task_params.items():
                    if param.grad is not None:
                        logging.info("update {} task specific param {} grad {}".format(task_idx,param_idx,param.grad))
                    else:
                        logging.info("{} task specific param {} grad is None".format(task_idx,param_idx))
                       
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss+=torch.sum(group_loss).item()
    
    return train_loss/len(dataloaders[0]),specific_parameters_set,task_groups
    
    
    
    
def train_woMeta(args,model,dataset,criterion,batchs):
    # Initialize a tensor to store all losses
    group_loss = torch.zeros(sum(args.divide_group.values()), device=args.device)
    index=0
    for group_id, group in enumerate(args.divide_group.keys()):
        for group_task in range(args.divide_group[group]):
            group_predictions, group_loss_cl_pois, group_loss_cl_users = model(dataset[group], batchs[group_id][group_task],index)
            loss_rec = criterion(group_predictions, batchs[group_id][group_task]["label"].to(args.device))
            base_loss = (1- args.lambda_cl) * loss_rec.mean() + args.lambda_cl * (group_loss_cl_users + group_loss_cl_pois)
            group_loss[index] = base_loss.mean()
            index+=1
    return group_loss
    

def eval_woMeta(args,model,test_dataset,batchs,Ks_list,pois_dict):
    
    recall_list = np.zeros(len(Ks_list))
    recall_array_dict={}
    mrr_array_dict={}
    predict_poi={}
    label_poi={}
    predict_distance_dict={}
    true_distance_dict={}
    for group in args.divide_group.keys():
        recall_array_dict[group] = np.zeros(shape=(args.divide_group[group], len(Ks_list)))
        mrr_array_dict[group] =  np.zeros(shape=(args.divide_group[group]))
        predict_poi[group]={}
        label_poi[group]={}
        predict_distance_dict[group]={}
        true_distance_dict[group]={}
        for i in range(args.divide_group[group]):
            predict_poi[group][i]=[]
            predict_distance_dict[group][i]=[]
            true_distance_dict[group][i]=[]
    
    index=0

    for group_id ,group in enumerate(args.divide_group.keys()):
            for group_task in range(args.divide_group[group]):
                with torch.no_grad():
                        y_pred_poi,loss_cl_poi,loss_cl_user = model(test_dataset[group], batchs[group_id][group_task],index)
                        label_poi=batchs[group_id][group_task]["label"].detach().cpu()
                        for k in Ks_list:
                            col_idx = Ks_list.index(k)
                            recall= batch_performance(y_pred_poi.detach().cpu(), label_poi, k)
                            recall_list[col_idx] += recall
                            recall_array_dict[group][group_task, col_idx] = recall
                        mrr=MRR_metric(y_pred_poi.detach().cpu(), label_poi)
                        mrr_array_dict[group][group_task] = mrr
                index+=1
    return recall_array_dict,mrr_array_dict




def eval(args,model,test_dataset,group_test_dataloader,Ks_list,poi_dict):
    
    dataloaders = list(group_test_dataloader.values())
    model.eval()
    iteration_num=len(dataloaders[0])
    
    full_recall_array_dict={}
    full_mrr_array_dict={}
    accuracy_l_dict={}
    mrr_l_dict={}
    predict_distance_dict={}
    true_distance_dict={}
    # result_dict={}
    for group in args.divide_group.keys():
        full_recall_array_dict[group]=np.zeros(shape=(iteration_num, args.divide_group[group], len(Ks_list)))
        full_mrr_array_dict[group]=np.zeros(shape=(iteration_num, args.divide_group[group]))
        accuracy_l_dict[group]= [TravellingMean() for _ in range(args.divide_group[group])]
        mrr_l_dict[group]=[TravellingMean() for _ in range(args.divide_group[group])]
        predict_distance_dict[group]={}
        true_distance_dict[group]={}
        for i in range(args.divide_group[group]):
            predict_distance_dict[group][i]=[]
            true_distance_dict[group][i]=[]

        
    for idx in range(iteration_num):
        batchs=[]
        
        for group_num in range(len(args.divide_group)):
            batchs.append(next(iter(dataloaders[group_num])))
            
        
        sample_recall_array_dict,sample_mrr_array_dict= eval_woMeta(args,model,test_dataset,batchs,Ks_list,poi_dict)
        
        
        for group_id,group in enumerate(args.divide_group.keys()):
        
            group_counts = [len(batchs[group_id][i]['group']) for i in range(args.divide_group[group])]
            for s in range(args.divide_group[group]):
                if sample_recall_array_dict[group][s, 1] is not None:
                    accuracy_l_dict[group][s].update(np.array(sample_recall_array_dict[group][s, 1]),group_counts[s])
                if sample_mrr_array_dict[group][s] is not None:
                    mrr_l_dict[group][s].update(np.array(sample_mrr_array_dict[group][s]),group_counts[s])
                
        
            
            for s in range(args.divide_group[group]):
                if sample_mrr_array_dict[group][s] is not None:
                    full_mrr_array_dict[group][idx,s]=sample_mrr_array_dict[group][s]
                else:
                    full_mrr_array_dict[group][idx,s]=None   
                for k in Ks_list:
                    col_idx = Ks_list.index(k)
                    if sample_recall_array_dict[group][s, col_idx] is not None:
                        full_recall_array_dict[group][idx,s,col_idx]=sample_recall_array_dict[group][s, col_idx]
                    else:
                        full_recall_array_dict[group][idx,s,col_idx]=None
    return full_recall_array_dict,full_mrr_array_dict,accuracy_l_dict ,mrr_l_dict