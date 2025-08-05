import logging
from dataset_woMeta import *
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
    # start_time = time.time()
    model.train()
    dataloaders = list(dataloader.values())  
    iteration_num=len(dataloaders[0])

    # Initialize task-specific parameters
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
    
                # 更新优化器，移除特定参数
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
                # 对每个任务的特定参数进行手动梯度更新
                for task_idx, task_params in model.task_specific_parameters.items():
                    for param_idx, param in task_params.items():
                        if param.grad is not None:
                            logging.info("update DCHL {} task specific param {} grad {}".format(task_idx,param_idx,param.grad))
                        else:
                            logging.info("DCHL {} task specific param {} grad is None".format(task_idx,param_idx))
                optimizer.step()
                optimizer.zero_grad()

              
        else:
            
            group_loss =train_woMeta(args,model,dataset,criterion,batchs)
            
            loss = torch.sum(group_loss)
            loss.backward()
            # 对每个任务的特定参数进行手动梯度更新
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
            
            # base_loss = loss_rec + args.lambda_cl * (group_loss_cl_users + group_loss_cl_pois)
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
    # result_dict={}
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
                        
                        
                        # user_seq=batchs[group_id][group_task]['user_seq'].detach().cpu()
                        # user_seq_len = batchs[group_id][group_task]['user_seq_len'].detach().cpu()
                        # group_prediction=y_pred_poi.detach().cpu()
                        label_poi=batchs[group_id][group_task]["label"].detach().cpu()
                        # user_ids=batchs[group_id][group_task]["user_idx"].detach().cpu()
                        
                        # count=group_prediction.size(0)
                        
                        
                        # predict_list=[]
                        # true_list=[]
                        
                        # for id in range(count):
                            
                        #     last_poi_indice=user_seq[id][user_seq_len[id]-1]
                        #     last_poi= pois_dict[pois_dict['Ven_Index']==last_poi_indice.item()]
                        #     last_lat = last_poi['Latitude'].values[0]
                        #     last_lon = last_poi['Longitude'].values[0]

                        #     target_poi= pois_dict[pois_dict['Ven_Index']==label_poi[id].item()]
                        #     target_lat = target_poi['Latitude'].values[0]
                        #     target_lon = target_poi['Longitude'].values[0]
                        #     ground_truth_distance= haversine_distance(last_lon,last_lat,target_lon,target_lat)
                        #     true_list.append(ground_truth_distance)
                            
                        #     y_pred_indice = group_prediction[id].topk(k=5).indices.tolist()
                            
                        #     for pred_ind in y_pred_indice: 
                        #         preidct_poi= pois_dict[pois_dict['Ven_Index']==pred_ind]
                        #         predict_lat = preidct_poi['Latitude'].values[0]
                        #         predict_lon = preidct_poi['Longitude'].values[0]
                        #         predict_distance= haversine_distance(last_lon,last_lat,predict_lon,predict_lat)
                        #         predict_list.append(predict_distance)
                        
                        # predict_distance_dict[group][group_task].extend(predict_list)
                        # true_distance_dict[group][group_task].extend(true_list)
                        #获取结束
                        
                        
                        for k in Ks_list:
                            col_idx = Ks_list.index(k)
                            recall= batch_performance(y_pred_poi.detach().cpu(), label_poi, k)
                            recall_list[col_idx] += recall
                            recall_array_dict[group][group_task, col_idx] = recall
                        mrr=MRR_metric(y_pred_poi.detach().cpu(), label_poi)
                        # mrr_array_dict[group][group_task] = mrr 
                        mrr_array_dict[group][group_task] = mrr
                       
                index+=1
    
    # for group in args.divide_group.keys():
    #     for group_task in range(args.divide_group[group]):
    #         logging.info("{} group {} predict_poi:  {}".format(group,group_task,predict_poi[group][group_task]))
    #         logging.info("{} group {} label_poi:    {}".format(group,group_task,label_poi[group][group_task]))
    #return recall_array_dict,mrr_array_dict,predict_distance_dict,true_distance_dict
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
                if sample_recall_array_dict[group][s, 1] is not None:# store sensitive-segregated values (update travelling means)
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