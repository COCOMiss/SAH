import argparse
import json
import time
import os
import logging
import yaml
import datetime
import torch.optim as optim
import random
from model_devide import *
from metrics import batch_performance, sample_performance
from train_nash import train,eval
from utils import *
from dataset import *

# clear cache
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True



# parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="NYC", help='NYC/TKY/Gowalla')
parser.add_argument('--seed', default=2023, help='Random seed')
parser.add_argument('--distance_threshold', default=2.5, type=float, help='distance threshold 2.5 or 0.25')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size')
parser.add_argument('--emb_dim', type=int, default=128, help='embedding size')
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--deviceID', type=int, default=1)
parser.add_argument('--lambda_cl', type=float, default=0.1, help='lambda of contrastive loss')
parser.add_argument('--num_mv_layers', type=int, default=3)
parser.add_argument('--num_geo_layers', type=int, default=3)
parser.add_argument('--num_di_layers', type=int, default=3, help='layer number of directed hypergraph convolutional network')
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--keep_rate', type=float, default=1, help='ratio of edges to keep')
parser.add_argument('--keep_rate_poi', type=float, default=1, help='ratio of poi-poi directed edges to keep')
parser.add_argument('--lr-scheduler-factor', type=float, default=0.1, help='Learning rate scheduler factor')
parser.add_argument('--save_dir', type=str, default="output/TKY/temp")

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

parser.add_argument('--divide_lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--finetune_batch_size', type=int, default=20, help='input batch size')

parser.add_argument('--divide_epoch', type=int, default=20)


parser.add_argument('--divide_group', type=str, default='{"User": 2,"Time" : 2,"POI":2}',help='JSON string for group dictionary')
parser.add_argument('--lrdecay', action='store', default=0.75, type=float, dest='lrdecay', help='learning rate decay')
parser.add_argument('--patience', type=int, default=10)

parser.add_argument('--time_embed_dim',type=int,default=32,help='Time embedding dimensions')


parser.add_argument('--transformer_nhid',type=int,default=1024,help='Hid dim in TransformerEncoder')
parser.add_argument('--transformer_nlayers',
                        type=int,
                        default=2,
                        help='Num of TransformerEncoderLayer')
parser.add_argument('--transformer_nhead',
                        type=int,
                        default=2,
                        help='Num of heads in multiheadattention')
parser.add_argument('--transformer_dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for transformer')

parser.add_argument('--accu_loss',
                        type=int,
                        default=20)
                         

args = parser.parse_args()

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# set device gpu/cpu
args.device = torch.device("cuda:{}".format(args.deviceID) if torch.cuda.is_available() else "cpu")


if args.divide_group:
    args.divide_group = json.loads(args.divide_group)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

current_save_dir = args.save_dir

# Setup logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=os.path.join(current_save_dir, f"log_training.txt"),
                    filemode='w+')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logging.getLogger('matplotlib.font_manager').disabled = True

# Save run settings
args_filename = args.dataset + '_args.yaml'
with open(os.path.join(current_save_dir, args_filename), 'w') as f:
    yaml.dump(vars(args), f, sort_keys=False)



def main():
    # Parse Arguments
    logging.info("1. Parse Arguments")
    logging.info(args)
    logging.info("device: {}".format(args.device))
   
    # Start GPU memory monitoring in background
    def monitor_gpu_memory():
        while True:
            try:
                import subprocess
                result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'])
                memory_used, memory_total = map(int, result.decode().strip().split(','))
                logging.info(f"GPU Memory Usage: {memory_used}/{memory_total} MB ({memory_used/memory_total*100:.1f}%)")
            except:
                pass
            time.sleep(5)  # Check every 5 seconds

    import threading
    monitor_thread = threading.Thread(target=monitor_gpu_memory, daemon=True)
    monitor_thread.start()
        
    if args.dataset=="NYC":
        NUM_USERS = 159
        NUM_POIS = 6870
        PADDING_IDX = NUM_POIS+1
 
    elif args.dataset=='TKY':
        NUM_USERS = 4928
        NUM_POIS = 17023
        PADDING_IDX = NUM_POIS+1

    elif args.dataset=='gowalla':
        NUM_USERS = 3996
        NUM_POIS = 9832
        PADDING_IDX = NUM_POIS+1

    # Load Dataset
    logging.info("2. Load Dataset")
    group_train_dataset={}
    group_test_dataset={}
    group_train_dataloader={}
    group_test_dataloader={}
    
    
    for group_mode in args.divide_group.keys():
        group_train_dataset[group_mode] = POIDataset(data_filename="datasets/{}/train.pkl".format(args.dataset),
                                data_time_filename="datasets/{}/train_time.pkl".format(args.dataset),
                                data_poitime_filename="datasets/{}/train_poi_time.pkl".format(args.dataset),
                                pois_coos_filename="datasets/{}/train_poi.pkl".format(args.dataset),
                                group_filename="datasets/{}/user_group.pkl".format(args.dataset),
                                last_poi_group_filename="datasets/{}/train_city_tag.pkl".format(args.dataset),
                                divide_group=group_mode,
                                num_users=NUM_USERS,
                                num_pois=NUM_POIS,
                                padding_idx=PADDING_IDX,
                                args=args,
                                device=args.device)
        
        group_test_dataset[group_mode] = POIDataset(data_filename="datasets/{}/test.pkl".format(args.dataset),
                                data_time_filename="datasets/{}/test_time.pkl".format(args.dataset),
                                data_poitime_filename="datasets/{}/test_poi_time.pkl".format(args.dataset),
                                pois_coos_filename="datasets/{}/test_poi.pkl".format(args.dataset),
                                group_filename="datasets/{}/user_group.pkl".format(args.dataset),
                                last_poi_group_filename="datasets/{}/test_city_tag.pkl".format(args.dataset),
                                divide_group=group_mode,
                                num_users=NUM_USERS,
                                num_pois=NUM_POIS,
                                padding_idx=PADDING_IDX,
                                args=args,
                                device=args.device)
        
         # 3. Construct DataLoader
        logging.info("3. Construct {} DataLoader".format(group_mode))
        
        
        group_train_dataloader[group_mode] = DataLoader(dataset= group_train_dataset[group_mode], batch_size=args.batch_size,
                                 collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX),
                                 sampler=BalancedSampler(group_train_dataset[group_mode].samples,batch_size=args.batch_size))
        
       
        
        group_test_dataloader[group_mode] = DataLoader(dataset=group_test_dataset[group_mode], batch_size=args.finetune_batch_size,
                                    collate_fn=lambda batch: collate_fn_4sq(batch, padding_value=PADDING_IDX),
                                    sampler=BalancedSampler(group_test_dataset[group_mode].samples,batch_size=args.finetune_batch_size))
    
    poi_dict=load_dict_from_pkl("datasets/{}/poi.pkl".format(args.dataset))

    # Load Model
    logging.info("4. Load Model")
    
    model = MSAHG(NUM_USERS, NUM_POIS, args, args.device)
    model.to(args.device)
    
    criterion = nn.CrossEntropyLoss(reduction='none',ignore_index=-1).to(args.device)
    optimizer = optim.Adam(params=list(model.parameters()),lr=args.lr)
    # Train
    logging.info("5. Start Training")
    Ks_list = [1, 5, 10, 20]
    final_results={}
    for group_mode, group_num in args.divide_group.items():
        final_results[group_mode]={}
        for s in range(group_num):
            final_results[group_mode][s]={"Rec1": 0.0, "Rec5": 0.0, "Rec10": 0.0, "Rec20": 0.0, "MRR" :0.0}

    
    saved_model_path = os.path.join(current_save_dir, "{}.pt".format(args.dataset))
    logging.info("Reload model")
    

    saved_group_result_path= os.path.join(current_save_dir, "group_result.txt")
    args.saved_model_path=saved_model_path
    
    group_num=sum(args.divide_group.values())
    group_weight=torch.ones(group_num)/group_num
    group_weight.to(args.device)

    ACC1_ACC20=0.0
    stop_counter=0
    specific_parameters_set=set()
    task_groups={}
    for epoch in range(args.num_epochs):
        
        if stop_counter>args.patience and epoch>args.divide_epoch+10:
            break    
    
        if epoch !=0: 
            checkpoint = torch.load(saved_model_path)
            model.load_state_dict(checkpoint['MSAHG_dict'])
            model.to(args.device)
            logging.info("Reload model")
            
        
        torch.cuda.empty_cache()
        start_time = time.time()
        
        logging.info('epoch: {} ; current group weights: {}'.format(epoch, to_np(group_weight)))
        if epoch == args.divide_epoch or epoch ==  args.divide_epoch+10:
            divide=True
        else:
            divide=False
        
        train_loss,specific_parameters_set,task_groups=train(args,model,group_train_dataset,criterion,group_train_dataloader,optimizer,specific_parameters_set,task_groups,divide)
        logging.info("Training finishes at this epoch. It takes {} min".format((time.time() - start_time) / 60))
        logging.info("Training Epoch {}/{} training loss: {}".format(epoch+1, args.num_epochs,train_loss))
        

        logging.info("Validation")
        valid_recall_array_dict,valid_mrr_array_dict,valid_accuracy_l_dict,valid_mrr_l_dict= eval(args,model,group_test_dataset,group_test_dataloader,Ks_list,poi_dict)
        logging.info("Validation finishes")
        logging.info("Validation results:")
        
        acc1_acc20=0.0 
        group_risks=[]
        mrr_reslut=0.0
        for group_mode in args.divide_group.keys():
            valid_recall_array=valid_recall_array_dict[group_mode]
            valid_mrr_array=valid_mrr_array_dict[group_mode]
            mrr_array = []
            for s in range(args.divide_group[group_mode]): 
                for k in Ks_list:
                    col_idx = Ks_list.index(k)   
                    recall = np.nanmean(valid_recall_array[:,s, col_idx])
                    logging.info("{} Group {}: Recall@{}: {:.4f}".format(group_mode,s,k, recall))
                    if k==1:
                        acc1_acc20+=recall
                    if k==20:
                        acc1_acc20+=recall*4
                mrr= np.nanmean(valid_mrr_array[:,s])
                mrr_array.append(mrr)
                mrr_reslut+=mrr
                group_risks.append(1-mrr)
                logging.info("{} Group {}: MRR : {:.4f}".format(group_mode,s, mrr))
                logging.info("\n")
            
        
        if epoch== args.divide_epoch or epoch== args.divide_epoch+10:
            state_dict = {
            'epoch': epoch,
            'MSAHG_dict': model.state_dict(),
            'task_groups':task_groups}
            
            torch.save(state_dict, saved_model_path)
            stop_counter=0
            optimizer_lr = args.lr
            optimizer = optim.Adam(params=list(model.parameters()),
                    lr=optimizer_lr)
        
        if epoch != args.divide_epoch and epoch!= args.divide_epoch+10:
            if  acc1_acc20 >= ACC1_ACC20  :
                ACC1_ACC20 = acc1_acc20
                logging.info("Update validation results and save model at epoch{}".format(epoch))
                
                state_dict = {
                'epoch': epoch,
                'MSAHG_dict': model.state_dict(),
                'task_groups':task_groups}
                    
                torch.save(state_dict, saved_model_path)
                stop_counter=0
                
            else:
                logging.info("At epoch {}  Stop Counter is {} ".format(epoch,stop_counter))
                stop_counter+=1
                
                optimizer_lr=args.lr*args.lrdecay
            
                optimizer = optim.Adam(params=list(model.parameters()),
                    lr=optimizer_lr)
        
                
    logging.info("Test")  
    if os.path.exists(saved_model_path):
        checkpoint = torch.load(saved_model_path)
        try:
            model.load_state_dict(checkpoint['MSAHG_dict'])
        except:
            task_groups = checkpoint['task_groups']
            model.add_task_specific_params(task_groups)
            model.load_state_dict(checkpoint['MSAHG_dict'])
        model.to(args.device)
        
        logging.info("Reload best training model")
   
    
    test_recall_array_dict,test_mrr_array_dict,test_recall_5_dict,test_mrr_dict= eval(args,model,group_test_dataset,group_test_dataloader,Ks_list,poi_dict)
    
        
    logging.info("Test finishes")
    logging.info("Test results:")
    
    for group_mode, group_num in args.divide_group.items():
        for s in range(group_num):
            mrr = np.nanmean(test_mrr_array_dict[group_mode][:,s])
            final_results[group_mode][s]["MRR"]=mrr
            for k in Ks_list:
                col_idx = Ks_list.index(k)
                
                recall =np.nanmean(test_recall_array_dict[group_mode][:, s, col_idx])
                
                if k == 1:
                    final_results[group_mode][s]["Rec1"] = recall

                elif k == 5:
                    final_results[group_mode][s]["Rec5"] =  recall

                elif k == 10:
                    final_results[group_mode][s]["Rec10"] = recall

                elif k == 20:
                    final_results[group_mode][s]["Rec20"] =  recall
                
                logging.info("{} Group {}: Recall@{}: {:.4f}".format(group_mode,s,k, recall))
            logging.info("\n") 
            logging.info("{} Group {}: MRR: {:.4f}".format(group_mode,s, mrr))
            logging.info("\n")
        logging.info("==================================\n\n")
    
        
        
    for group_mode, group_num in args.divide_group.items():
        for s in range(group_num):
            logging.info("{} Group {} query user count is {}".format(group_mode,s,len(group_test_dataset[group_mode].samples[s])))
    
   
    with open(saved_group_result_path, "w") as file:
        for group_mode, group_result in final_results.items():
            for group_key, group_metrics in group_result.items():
                for metric_key, metric_value in group_metrics.items():
                    file.write(f"{group_mode} {group_key}: {metric_key}: {metric_value}\n")
                
if __name__ == '__main__':
    main()

