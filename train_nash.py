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
    # node_trans_dict_model.train()
    # node_trans_dict_model.to_train()
    # Assuming group_support_dataloader is a dictionary of DataLoaders
    # Get the DataLoaders from the dictionary
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
            
            # group_loss =train_woMeta(args,node_attn_model,None,dataset,criterion,batchs)
       
            if i<=args.accu_loss:
                devide_loss=group_loss+devide_loss
            if i==args.accu_loss:

                # 获取新的任务特定参数
                original_len=len(specific_parameters_set)
                # original_attn_len=len(attn_specific_parameters_set)
                task_groups,specific_parameters_set=method.task_specific_params(losses=devide_loss,
                        model=model,
                        specific_parameters_set=specific_parameters_set)
                # attn_task_groups,attn_specific_parameters_set=method.task_specific_params(losses=devide_loss,
                #         model=node_attn_model,
                #         specific_parameters_set=attn_specific_parameters_set)
                
                
                updated_len=len(specific_parameters_set)
                # attn_updated_len=len(attn_specific_parameters_set)
    
                # 更新优化器，移除特定参数
                if updated_len-original_len > 0:
                    if updated_len-original_len > 0:
                        model.add_task_specific_params(task_groups)
                    # if attn_updated_len-original_attn_len > 0:
                    #     node_attn_model.add_task_specific_params(attn_task_groups)
                    
                    logging.info('Create new optimizer')
                    # 创建新的优化器，只包含共享参数          
                    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                    
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
                            
                # for task_idx, task_params in node_attn_model.task_specific_parameters.items():
                #     for param_idx, param in task_params.items():
                #         if param.grad is not None:
                #             logging.info("update attn model {} task specific param {} grad {}".format(task_idx,param_idx,param.grad))
                #         else:
                #             logging.info("attn model {} task specific param {} grad is None".format(task_idx,param_idx))
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
            # if index in task_specific_params:
            #     specific_params_dict=task_specific_params[index]
            # else:
            #     specific_params_dict=None
                
            
            group_predictions, group_loss_cl_pois, group_loss_cl_users = model(dataset[group], batchs[group_id][group_task],index)
            
            # loss_rec=0.0
            # poi_group_losses = []  # 存储每个POI组的loss
            
            # if node_trans_dict_model is not None:
            #     for poi_group_index in range(args.divide_group['POI']):
            #         poi_group_mask= batchs[group_id][group_task]['poi_group']==poi_group_index
            #         if torch.any(poi_group_mask):
            #             group_adjusted_predictions=node_trans_dict_model(group_predictions[poi_group_mask],poi_group_index, 
            #                                                           batchs[group_id][group_task]['user_seq'][poi_group_mask], 
            #                                                           batchs[group_id][group_task]['user_seq_len'][poi_group_mask])
            #             loss_rec_poi_group_temp = criterion(group_adjusted_predictions, 
            #                                               batchs[group_id][group_task]["label"][poi_group_mask].to(args.device))
            #             poi_group_losses.extend(loss_rec_poi_group_temp)
            
            # 计算所有POI组loss的均值
            # if poi_group_losses:
            #     loss_rec = torch.stack(poi_group_losses).mean()
            
            loss_rec = criterion(group_predictions, batchs[group_id][group_task]["label"].to(args.device))
            base_loss = loss_rec.mean() + args.lambda_cl * (group_loss_cl_users + group_loss_cl_pois)
            
            # base_loss = loss_rec + args.lambda_cl * (group_loss_cl_users + group_loss_cl_pois)
            group_loss[index] = base_loss.mean()
            index+=1
    return group_loss
    

def eval_woMeta(args,model,test_dataset,batchs,Ks_list,pois_dict):
    
    
    recall_array_dict={}
    mrr_array_dict={}
    predict_poi={}
    label_poi={}
    test_category_array_dict={}
    for group in args.divide_group.keys():
        recall_array_dict[group] = np.zeros(shape=(args.divide_group[group], len(Ks_list)))
        mrr_array_dict[group] =  np.zeros(shape=(args.divide_group[group]))
        predict_poi[group]={}
        label_poi[group]={}
        test_category_array_dict[group]={}
        for i in range(args.divide_group[group]):
            predict_poi[group][i]=[]
            test_category_array_dict[group][i]=[]
    
    index=0
    
    for group_id ,group in enumerate(args.divide_group.keys()):
            for group_task in range(args.divide_group[group]):
                
                with torch.no_grad():
       
                        result=0.0
    
                        y_pred_poi,loss_cl_poi,loss_cl_user = model(test_dataset[group], batchs[group_id][group_task],index)
                        
                        # batch_size=batchs[group_id][group_task]['user_seq_len'].shape
                        label_poi=batchs[group_id][group_task]["label"].detach().cpu()
                        
                       
                        # 获取category
                        group_prediction=y_pred_poi.detach().cpu()
                        count=group_prediction.size(0)
                        category_list=[]
                        for id in range(count):
                            y_pred_indice = group_prediction[id].topk(k=5).indices.tolist()
                            for pred_ind in y_pred_indice:
                                category=pois_dict[pois_dict['Ven_Index']==pred_ind]['Main Category'].values[0]
                                category_list.append(category)
                        test_category_array_dict[group][group_task].extend(category_list)
                        ##获取结束
                        
                        
                        for k in Ks_list:
                            col_idx = Ks_list.index(k)
                            recall= batch_performance(y_pred_poi.detach().cpu(), label_poi, k)
                            # result+=recall
                            # recall_array_dict[group][group_task, col_idx] = recall
                            recall_array_dict[group][group_task, col_idx] = recall
                        mrr=MRR_metric(y_pred_poi.detach().cpu(), label_poi)
                        # mrr_array_dict[group][group_task] = mrr 
                        mrr_array_dict[group][group_task] = mrr 
            
                        
                        
                        
                        # if node_trans_dict_model is not None:
                        #     for poi_group_index in range(args.divide_group['POI']):
                        #         poi_group_mask= batchs[group_id][group_task]['poi_group']==poi_group_index
                        #         # sample_count  = torch.sum(poi_group_mask).item()
                        #         if torch.any(poi_group_mask):
                        #             group_adjusted_predictions=node_trans_dict_model(y_pred_poi[poi_group_mask],poi_group_index, 
                        #                                                         batchs[group_id][group_task]['user_seq'][poi_group_mask], 
                        #                                                         batchs[group_id][group_task]['user_seq_len'][poi_group_mask])
                                    
                        #             label_poi=batchs[group_id][group_task]["label"][poi_group_mask].detach().cpu()
                                    
                        #             for k in Ks_list:
                        #                 col_idx = Ks_list.index(k)
                        #                 recall= batch_performance(group_adjusted_predictions.detach().cpu(), label_poi, k)
                        #                 # result+=recall
                        #                 # recall_array_dict[group][group_task, col_idx] = recall
                        #                 recall_array_dict[group][group_task, col_idx] += recall
                        #             mrr=MRR_metric(group_adjusted_predictions.detach().cpu(), label_poi)
                        #             # mrr_array_dict[group][group_task] = mrr 
                        #             mrr_array_dict[group][group_task] += mrr 
                                          
                        # for k in Ks_list:
                        #     col_idx = Ks_list.index(k)
                        #     recall_array_dict[group][group_task, col_idx] = recall_array_dict[group][group_task, col_idx]/batch_size
                        # mrr_array_dict[group][group_task] = mrr_array_dict[group][group_task]/batch_size
                        
                             
                        # adjust_group_predictions=node_trans_dict_model(y_pred_poi, batchs[group_id][group_task]['user_seq'], batchs[group_id][group_task]['user_seq_len'])
                                        
                        # for i in  range(adjust_group_predictions.size(0)):
                        #     y_pred_indices = adjust_group_predictions[i].topk(k=1).indices.tolist()
                        #     predict_poi[group][group_task].append(y_pred_indices)
                        
                        # label_poi[group][group_task]=batchs[group_id][group_task]["label"].detach().cpu()
                        
                        
                        # for k in Ks_list:
                        #     col_idx = Ks_list.index(k)
                        #     recall= batch_performance(adjust_group_predictions.detach().cpu(), batchs[group_id][group_task]["label"].detach().cpu(), k)
                        #     result+=recall
                        #     recall_array_dict[group][group_task, col_idx] = recall
                        # mrr=MRR_metric(adjust_group_predictions.detach().cpu(), batchs[group_id][group_task]["label"].detach().cpu())
                        # mrr_array_dict[group][group_task] = mrr
                index+=1
    
    # for group in args.divide_group.keys():
    #     for group_task in range(args.divide_group[group]):
    #         logging.info("{} group {} predict_poi:  {}".format(group,group_task,predict_poi[group][group_task]))
    #         logging.info("{} group {} label_poi:    {}".format(group,group_task,label_poi[group][group_task]))
    return recall_array_dict,mrr_array_dict,test_category_array_dict       



def eval(args,model,test_dataset,group_test_dataloader,Ks_list,poi_dict):
    
    
    # Get the DataLoaders from the dictionary
    dataloaders = list(group_test_dataloader.values())
    model.eval()
    # node_trans_dict_model.eval()
    # node_trans_dict_model.to_eval()
    
    # if len(group_support_dataloader)>1:
    iteration_num=len(dataloaders[0])
    
    full_recall_array_dict={}
    full_mrr_array_dict={}
    accuracy_l_dict={}
    mrr_l_dict={}
    category_l_dict={}
    for group in args.divide_group.keys():
        full_recall_array_dict[group]=np.zeros(shape=(iteration_num, args.divide_group[group], len(Ks_list)))
        full_mrr_array_dict[group]=np.zeros(shape=(iteration_num, args.divide_group[group]))
        accuracy_l_dict[group]= [TravellingMean() for _ in range(args.divide_group[group])]
        mrr_l_dict[group]=[TravellingMean() for _ in range(args.divide_group[group])]
        category_l_dict[group]={}
        for i in range(args.divide_group[group]):
            category_l_dict[group][i]=[]

        
    for idx in range(iteration_num):
        batchs=[]
        
        for group_num in range(len(args.divide_group)):
            batchs.append(next(iter(dataloaders[group_num])))
            
        
        sample_recall_array_dict,sample_mrr_array_dict,sample_category_array_dict= eval_woMeta(args,model,test_dataset,batchs,Ks_list,poi_dict)
        
        
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
                
                if sample_category_array_dict[group][s] is not None:
                    category_l_dict[group][s].extend(sample_category_array_dict[group][s])
                
                
                for k in Ks_list:
                    col_idx = Ks_list.index(k)
                    if sample_recall_array_dict[group][s, col_idx] is not None:
                        full_recall_array_dict[group][idx,s,col_idx]=sample_recall_array_dict[group][s, col_idx]
                    else:
                        full_recall_array_dict[group][idx,s,col_idx]=None
    

    return full_recall_array_dict,full_mrr_array_dict,accuracy_l_dict ,mrr_l_dict,category_l_dict

