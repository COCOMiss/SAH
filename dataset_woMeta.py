# coding=utf-8
"""
@author: Yantong Lai
@paper: [24 SIGIR] Disentangled Contrastive Hypergraph Learning for Next POI Recommendation
"""

import os
from utils import *
from torch.utils.data import Dataset, DataLoader,Sampler
from torch.nn.utils.rnn import pad_sequence
import random


class POIDataset(Dataset):
    def __init__(self, data_filename,data_time_filename,data_poitime_filename, pois_coos_filename,group_filename,last_poi_group_filename,divide_group, num_users, num_pois, padding_idx, args, device):

        # get all sessions and labels
        self.data = load_list_with_pkl(data_filename)  # data = [sessions_dict, labels_dict]
        self.time_data = load_list_with_pkl(data_time_filename)  # data = [user_time_dict, user_last_time_dict]
        self.poitime_session_data=load_dict_from_pkl(data_poitime_filename) # data = user_poi_time_dict
        # self.last_poi_group_data=load_dict_from_pkl(last_poi_group_filename) # data = user_last_poi_group_dict
        
        self.sessions_dict = self.data[0]  # poiID starts from 0
        self.labels_dict = self.data[1]
        
        
        self.time_session_dict=self.time_data[0]
        self.last_time_dict=self.time_data[1]
        self.time_labels=self.time_data[2]
        
        self.pois_coos_dict = load_dict_from_pkl(pois_coos_filename)
        self.user_label_dict= load_dict_from_pkl(group_filename)
        self.args=args
                

        # definition
        self.num_users = len(self.user_label_dict)
        self.num_pois = num_pois
        # self.num_sessions = get_num_sessions(self.sessions_dict)
        self.padding_idx = -1
        self.distance_threshold = args.distance_threshold
        self.keep_rate = args.keep_rate
        self.device = device
        self.divide_group=divide_group

        # get user's trajectory, reversed trajectory and its length
        self.users_trajs_dict, self.users_trajs_labels_dict, self.users_trajs_lens_dict,self.users_trajs_lastpoi_group_dict = get_user_traj(self.sessions_dict,self.labels_dict) 
        self.users_traj_time_dict, self.users_traj_labels_time_dict,self.users_traj_time_lens_dict,_ = get_user_traj(self.time_session_dict, self.time_labels)
        # self.last_time_dict=get_traj_lasttime(self.time_session_dict)
        
        self.group_num={}
        self.group_number=args.divide_group[divide_group]
        for s in range(self.group_number):
            self.group_num[s] = 0
        
        if divide_group=='Time':
            self.time_label_dict={i: 0 for i in range(48)} ####工作日为group 0 休息日 为group1
            self.time_label_dict.update({i: 1 for i in range(48,96)}) 
            for last_time_sessions in self.last_time_dict.values():
                for last_time in last_time_sessions:
                    ###这里需要添加统计time label的代码
                    self.group_num[self.time_label_dict[last_time]]+=1
        elif divide_group=='User':
            for user_traj in self.users_trajs_dict.values():
                user_idx=user_traj[0]
                self.group_num[self.user_label_dict[user_idx]]+=1
        elif divide_group=='POI':
            
            for last_poi in self.users_trajs_lastpoi_group_dict.values():
                self.group_num[int(self.pois_coos_dict[last_poi][0][2])]+=1
            # for group_sessions in self.last_poi_group_data.values():
            #     for group in group_sessions:
            #         self.group_num[group]+=1
            
        dataset_mode = data_filename.split('/')[-1].split('.')[0]
        dataset_name=  data_filename.split('/')[-2]
        if os.path.exists("datasets/{}/{}_poi_geo_graph_matrix.pkl".format(dataset_name,dataset_mode)):
            # 计算时间太长 直接存储使用
            self.poi_geo_graph_matrix = load_dict_from_pkl("datasets/{}/{}_poi_geo_graph_matrix.pkl".format(dataset_name,dataset_mode))  # Load the existing graph matrix
        else:
            # calculate poi-poi haversine distance and generate geographical adjacency matrix
            self.poi_geo_adj = gen_poi_geo_adj(num_pois, self.pois_coos_dict, self.distance_threshold)   # csr_matrix
            self.poi_geo_graph_matrix = normalized_adj(adj=self.poi_geo_adj, is_symmetric=False)
            save_dict_to_pkl("datasets/{}/{}_poi_geo_graph_matrix.pkl".format(dataset_name,dataset_mode), self.poi_geo_graph_matrix)
        self.poi_geo_graph = transform_csr_matrix_to_tensor(self.poi_geo_graph_matrix).to(device)
        
        
        ##开始建图 (time-poi，poi-time)  (time-user,user-time) (poi-user,user-poi) (poi-src,poi-tar)
        ## (poi-user,user-poi); 根据user 进行分类
        ## time POI 建图 得到归一化后的 time-poi 和 poi-time
        if divide_group=='Time':
            self.HG_tp,self.HG_pt=self.get_graph(dataset_name,file_name="time_{}_H_tp.pkl".format(dataset_mode),graph_mode="time_poi",divide_group="time_poi",group_label_dict=self.time_label_dict)
            ## time user 建图 得到归一化后的 time-user 和 user-time
            self.HG_tu,self.HG_ut=self.get_graph(dataset_name,file_name="time_{}_H_tu.pkl".format(dataset_mode),graph_mode="time_user",divide_group="time_user",group_label_dict=self.time_label_dict)
    
            self.HG_pu,self.HG_up=self.get_graph(dataset_name,file_name="{}_H_pu.pkl".format(dataset_mode),graph_mode="poi_user")
            ## POI POI trans 建图 得到归一化后的 poi-src 和 user-tar
            self.HG_poi_src,self.HG_poi_tar=self.get_graph(dataset_name,file_name="{}_H_poi_src.pkl".format(dataset_mode),graph_mode="tran_poi")
            
        elif divide_group=='User':
            
            self.HG_tp,self.HG_pt=self.get_graph(dataset_name,file_name="{}_H_tp.pkl".format(dataset_mode),graph_mode="time_poi")
            ## time user 建图 得到归一化后的 time-user 和 user-time
            self.HG_tu,self.HG_ut=self.get_graph(dataset_name,file_name="{}_H_tu.pkl".format(dataset_mode),graph_mode="time_user")
            ## POI user 建图 得到归一化后的 poi-user 和 user-poi
            self.HG_pu,self.HG_up=self.get_graph(dataset_name,file_name="user_{}_H_pu.pkl".format(dataset_mode),graph_mode="poi_user",divide_group="poi_user",group_label_dict=self.user_label_dict)
            ## POI POI trans 建图 得到归一化后的 poi-src 和 user-tar
            self.HG_poi_src,self.HG_poi_tar=self.get_graph(dataset_name,file_name="{}_H_poi_src.pkl".format(dataset_mode),graph_mode="tran_poi",divide_group="User")
            
        else:
            self.HG_tp,self.HG_pt=self.get_graph(dataset_name,file_name="{}_H_tp.pkl".format(dataset_mode),graph_mode="time_poi")
            ## time user 建图 得到归一化后的 time-user 和 user-time
            self.HG_tu,self.HG_ut=self.get_graph(dataset_name,file_name="{}_H_tu.pkl".format(dataset_mode),graph_mode="time_user")
            ## POI user 建图 得到归一化后的 poi-user 和 user-poi
            self.HG_pu,self.HG_up=self.get_graph(dataset_name,file_name="{}_H_pu.pkl".format(dataset_mode),graph_mode="poi_user")
            ## POI POI trans 建图 得到归一化后的 poi-src 和 user-tar
            self.HG_poi_src,self.HG_poi_tar=self.get_graph(dataset_name,file_name="{}_H_poi_src.pkl".format(dataset_mode),graph_mode="tran_poi")
            
        self.split_samples(divide_group)
        


    def __len__(self):
        # return self.num_users
        return len(self.users_trajs_dict)  # 直接返回字典的长度

    def __getitem__(self, idx):
        
        # samples=[]
        # for idxs in batch_idxs:
        # for idx in idxs:
        traj_idx = str(list(self.users_trajs_dict.keys())[idx])
        user_idx= self.users_trajs_dict[traj_idx][0]
        traj_seq = self.users_trajs_dict[traj_idx][1]
        time_seq= self.users_traj_time_dict[traj_idx][1]
        last_time=time_seq[-1]
        traj_seq_len = self.users_trajs_lens_dict[traj_idx]
        user_seq_mask = [1] * traj_seq_len
       
       
        traj_labels=self.users_trajs_labels_dict[traj_idx]
        # traj_labels = self.labels_dict[traj_idx]
        
        time_labels=self.users_traj_labels_time_dict[traj_idx]
        poi_group=int(self.pois_coos_dict[traj_seq[-1]][0][2])
        
        if self.divide_group=='User':
            group=self.user_label_dict[user_idx]
        elif self.divide_group=='Time':
            group=self.time_label_dict[last_time]
        elif self.divide_group=='POI':
            # traj_id = traj_idx.split('_')[-1]
            # group=self.last_poi_group_data[str(user_idx)][int(traj_id)] 
            group= int(self.pois_coos_dict[traj_seq[-1]][0][2])
        

        sample = {
            "user_idx": torch.tensor(int(user_idx)).to(self.device),
            "user_seq": torch.tensor(traj_seq).to(self.device),
            "time_seq":  torch.tensor(time_seq).to(self.device),
            "user_seq_len": torch.tensor(traj_seq_len).to(self.device),
            "user_seq_mask": torch.tensor(user_seq_mask).to(self.device),
            "label": torch.tensor(traj_labels).to(self.device),
            "time_label":  torch.tensor(time_labels).to(self.device),
            "last_time": torch.tensor(last_time).to(self.device),
            "poi_group":torch.tensor(poi_group).to(self.device),
            "group":torch.tensor(group).to(self.device)
        }
        # samples.append(sample)

        return sample
    
    
    
    
    
    
    def split_samples(self,divide_group=None):
        """Split the dataset into each group."""
        
        self.samples = {group: [] for group in range(self.group_number)}
       
        
        
        if divide_group=="User":
             for idx,traj_id in enumerate(self.users_trajs_dict.keys()):
                user_idx = self.users_trajs_dict[traj_id][0]
                group=self.user_label_dict[str(user_idx)]
                self.samples[group].append(idx)
               
        elif divide_group=="Time":
            for idx ,traj_time in enumerate(self.users_traj_time_dict.values()):
                times=traj_time[1]
                group=self.time_label_dict[times[-1]]
                self.samples[group].append(idx)
              
        elif divide_group=="POI":
            for idx,last_poi in enumerate(self.users_trajs_lastpoi_group_dict.values()):
                last_poi_group=int(self.pois_coos_dict[last_poi][0][2])
                # self.group_num[int(self.pois_coos_dict[last_poi][0][2])]+=1
                self.samples[last_poi_group].append(idx)
    
    
    
    
    def get_graph(self,dataset_name,file_name,graph_mode,divide_group=None,group_label_dict=None):
        
        ## time POI 建图
        if os.path.exists("datasets/{}/{}.pkl".format(dataset_name,file_name)):
            # 计算时间太长 直接存储使用
            H_rc = load_dict_from_pkl("datasets/{}/{}.pkl".format(dataset_name,file_name))  # Load the existing graph matrix
        else:
            if graph_mode==divide_group :
                if graph_mode=='time_poi':
                    # generate poi-session incidence matrix, its degree and hypergraph
                    # H_tp dict [T, P]   
                    H_rc = gen_sparse_H_time_poi(self.poitime_session_data, time_category=96, num_poi=self.num_pois,group_dict=group_label_dict,group_num=self.args.divide_group['Time'])
                elif graph_mode=='time_user':
                    ##先不对time进行分组，group——num 为1
                    H_rc = gen_sparse_H_time(self.time_session_dict, time_category=96, num_users=self.num_users,time_group_dict=group_label_dict,group_num=self.args.divide_group['Time'])    # H_tu dict [T, U]
                elif graph_mode=='poi_user':
                    # generate poi-session incidence matrix, its degree and hypergraph
                    H_rc = gen_sparse_H_user(self.sessions_dict, self.num_pois,  self.num_users,group_label_dict,self.args.divide_group['User'])    # H_pu dict [L, U]
                    
                for key,H in H_rc.items():
                    H_rc[key]=csr_matrix_drop_edge(H, self.args.keep_rate)
            else:
                if graph_mode=='time_poi':
                    # generate poi-session incidence matrix, its degree and hypergraph
                    # H_tp dict [T, P]   
                    H_rc = gen_sparse_H_time_poi(self.poitime_session_data, time_category=96, num_poi=self.num_pois)
                elif graph_mode=='time_user':
                    ##先不对time进行分组，group——num 为1
                    H_rc = gen_sparse_H_time(self.time_session_dict, time_category=96, num_users=self.num_users)    # H_tu dict [T, U]
                elif graph_mode=='poi_user':
                    # generate poi-session incidence matrix, its degree and hypergraph
                    H_rc = gen_sparse_H_user(self.sessions_dict, self.num_pois,  self.num_users)    # H_pu dict [L, U]
                elif graph_mode=='tran_poi':
                    # generate directed poi-poi hypergraph
                    H_rc = gen_sparse_directed_H_poi(self.users_trajs_dict, self.num_pois)    # [L, L]
                 # drop edge on csr_matrix H_pu
                H_rc=csr_matrix_drop_edge(H_rc, self.args.keep_rate)
            
            # self.H_pu = csr_matrix_drop_edge(self.H_pu, args.keep_rate)
            save_dict_to_pkl("datasets/{}/{}.pkl".format(dataset_name,file_name), H_rc)
                 
        if graph_mode==divide_group:
            ## 归一化，并得到 C R 图
            HG_rc={}
            HG_cr={} 
            # get degree of H_rc
            for key,H in H_rc.items():
                Deg_H_rc = get_hyper_deg(H)  # [T, T]
                # normalize poi-user hypergraph
                HG_rc[key]= Deg_H_rc * H   # [T, P]
                HG_rc[key] = transform_csr_matrix_to_tensor(HG_rc[key]).to(self.device)
                
                # generate session-poi incidence matrix, its degree and hypergraph
                Deg_H_cr = get_hyper_deg(H.T)    # [P, P]
                HG_cr[key] = Deg_H_cr * H.T    # [P, T]
                HG_cr[key] = transform_csr_matrix_to_tensor(HG_cr[key]).to(self.device)
        else:
            Deg_H_rc = get_hyper_deg(H_rc)  # [T, T]
            # normalize poi-user hypergraph
            HG_rc= Deg_H_rc * H_rc  # [T, P]
            HG_rc = transform_csr_matrix_to_tensor(HG_rc).to(self.device)
            
            # generate session-poi incidence matrix, its degree and hypergraph
            Deg_H_cr = get_hyper_deg(H_rc.T)    # [P, P]
            HG_cr = Deg_H_cr * H_rc.T   # [P, T]
            HG_cr = transform_csr_matrix_to_tensor(HG_cr).to(self.device)
        
        return HG_rc,HG_cr
    
    


class POIPartialDataset(Dataset):
    def __init__(self, full_dataset, user_indices):
        self.data = [full_dataset[i] for i in user_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]





def collate_fn_4sq(batch, padding_value=-1):
    """
    Pad sequence in the batch into a fixed length
    batch: list obj
    """
    
    mid_index = len(batch) // 2
    new_batch=[batch[:mid_index],batch[mid_index:]]
    collate_samples=[]
    for items in new_batch:
        batch_user_idx = []
        batch_user_seq = []
        batch_time_seq=[]
        batch_user_seq_len = []
        batch_user_seq_mask = []
        batch_label = []
        batch_time_label=[]
        batch_last_time=[]
        batch_poi_group=[]
        batch_group=[]
        for item in items:
            batch_user_idx.append(item["user_idx"])
            batch_user_seq_len.append(item["user_seq_len"])
            batch_label.append(item["label"])
            batch_time_label.append(item["time_label"])
            batch_last_time.append(item["last_time"])
            batch_poi_group.append(item["poi_group"])
            batch_group.append(item["group"])
            batch_user_seq.append(item["user_seq"])
            batch_time_seq.append(item["time_seq"])
            batch_user_seq_mask.append(item["user_seq_mask"])
            
         # pad seq and rev seq
        pad_user_seq = pad_sequence(batch_user_seq, batch_first=True, padding_value=-1)
        pad_time_seq= pad_sequence(batch_time_seq, batch_first=True, padding_value=-1)
        # pad_user_labels=pad_sequence(batch_label, batch_first=True, padding_value=-1)
        # pad_time_labels=pad_sequence(batch_time_label, batch_first=True, padding_value=-1)
        pad_user_seq_mask = pad_sequence(batch_user_seq_mask, batch_first=True, padding_value=-1)

        # stack list obj to a torch.tensor
        batch_user_idx = torch.stack(batch_user_idx)
        batch_user_seq_len = torch.stack(batch_user_seq_len)
        batch_label = torch.stack(batch_label)
        batch_time_label = torch.stack(batch_time_label)
        batch_last_time=torch.stack(batch_last_time)
        batch_poi_group = torch.stack(batch_poi_group)
        batch_group=torch.stack(batch_group)
        
        

        collate_sample = {
            "user_idx": batch_user_idx,
            "user_seq": pad_user_seq,
            "time_seq":pad_time_seq,
            "user_seq_len": batch_user_seq_len,
            "user_seq_mask": pad_user_seq_mask,
            "label": batch_label,
            "time_label": batch_time_label,
            "last_time":batch_last_time,
            "poi_group": batch_poi_group,
            "group":batch_group
        }
        collate_samples.append(collate_sample)

    return collate_samples


class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, samples, batch_size):
        self.samples = samples
        self.batch_size = batch_size

    def __iter__(self):
        # Calculate the number of samples to draw from each group
        num_samples_per_group = self.batch_size//len(self.samples)
        # batch_indices=[]
        indices = []
        num_iter=sum(len(group) for group in self.samples.values()) // self.batch_size
        for i in range(num_iter):
            # Sample from each group
            for i, group_samples in enumerate(self.samples.values()):
                sampled_indices = random.sample(group_samples, min(num_samples_per_group, len(group_samples)))
                indices.extend(sampled_indices)
        # batch_indices.append(indices)

        # Shuffle the indices to ensure randomness
        # random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        # length=sum(len(group) for group in self.samples.values()) // self.batch_size
        return sum(len(group) for group in self.samples.values())