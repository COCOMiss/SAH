from collections import defaultdict
import pickle
import numpy as np
from math import radians, cos, sin, asin, sqrt
import scipy.sparse as sp
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
import dgl



class TravellingMean:
    def __init__(self):
        self.count = 0
        self._mean= 0

    @property
    def mean(self):
        return self._mean

    def update(self, val,count=0):
        if val is None:
            return
        if count==0:
            self.count+=val.shape[0]
            self._mean += ((np.mean(val)-self._mean)*val.shape[0])/self.count
        else:
            self.count+=count
            self._mean += ((np.mean(val)-self._mean)*count)/self.count

def get_unique_seq(sessions_list):
    """Get unique POIs in the sequence"""
    seq_list = []
    for session in sessions_list:
        for poi in session:
            if poi in seq_list:
                continue
            else:
                seq_list.append(poi)

    return seq_list

def to_np(t):
    return t.cpu().detach().numpy()


def to_catgorical(group_attr,args):
    y = np.array(group_attr, dtype='int')
    y = y.ravel()
    n = y.shape[0]
    categorical = np.zeros((n, args.n_sensitive), dtype=float)
    categorical[np.arange(n), y] = 1
    return categorical

def pareto_check(table_results, eps_pareto=0):

    #table_results dimension: objectives x iterations
    pareto = np.zeros([table_results.shape[1]])
    if table_results.shape[1] == 1:
        pareto[0] = True
        # print('one')
    else:
        for i in np.arange(table_results.shape[1]):
            dif_risks = table_results[:, i][:, np.newaxis] - table_results  #(dim: objectives x iterations)
            dif_risks = dif_risks < eps_pareto  #evaluates if item iteration i dominates some coordinate, 1 indicates dominance
            dif_risks[:,i] = 1 #we would like to have a row full of ones
            dif_risks = np.sum(dif_risks,axis = 0) #sum on objectives
            pareto[i] = (np.prod(dif_risks) > 0) #prod of iterations (I cannot have an iteration that = 0 when sum all objectives differences

    if np.sum(pareto) == 0: #if all vectors are the same basically...
        ix_p = np.argmin(np.sum(table_results,axis=0))
        pareto[ix_p] = 1
    return pareto


def get_unique_seqs_for_sessions(sessions_dict):
    """Get unique seq for each session"""
    seqs_dict = {}
    seqs_lens_dict = {}
    for key, value in sessions_dict.items():
        seqs_dict[key] = get_unique_seq(value)
        seqs_lens_dict[key] = len(get_unique_seq(value))

    return seqs_dict, seqs_lens_dict


def get_seqs_for_sessions(sessions_dict, padding_idx, max_seq_len):
    seqs_dict = {}
    seqs_lens_dict = {}
    reverse_seqs_dict = {}
    for key, sessions in sessions_dict.items():
        temp = []
        for session in sessions:
            temp.extend(session)
        if len(temp) >= max_seq_len:
            temp = temp[-max_seq_len:]
            temp_rev = temp[::-1]
            seqs_dict[key] = temp
            reverse_seqs_dict[key] = temp_rev
            seqs_lens_dict[key] = max_seq_len
        else:
            temp_new = temp + [padding_idx] * (max_seq_len - len(temp))
            temp_rev = temp[::-1] + [padding_idx] * (max_seq_len - len(temp))
            seqs_dict[key] = temp_new
            reverse_seqs_dict[key] = temp_rev
            seqs_lens_dict[key] = len(temp)

    return seqs_dict, reverse_seqs_dict, seqs_lens_dict


def save_list_with_pkl(filename, list_obj):
    with open(filename, 'wb') as f:
        pickle.dump(list_obj, f)


def load_list_with_pkl(filename):
    with open(filename, 'rb') as f:
        list_obj = pickle.load(f)
    return list_obj


def save_dict_to_pkl(pkl_filename, dict_pbj):
    with open(pkl_filename, 'wb') as f:
        pickle.dump(dict_pbj, f)


def load_dict_from_pkl(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        dict_obj = pickle.load(f)

    return dict_obj


def get_num_sessions(sessions_dict):
    num_sessions = 0
    for value in sessions_dict.values():
        num_sessions += len(value)

    return num_sessions


def get_user_complete_traj(sessions_dict):
    """Get each user's complete trajectory from her sessions"""
    users_trajs_dict = {}
    users_trajs_lens_dict = {}
    for userID, sessions in sessions_dict.items():
        traj = []
        for session in sessions:
            traj.extend(session)
        users_trajs_dict[userID] = traj
        users_trajs_lens_dict[userID] = len(traj)

    return users_trajs_dict, users_trajs_lens_dict

def get_user_traj(sessions_dict,label_dict):
    """Get each user's complete trajectory from her sessions"""
    users_trajs_dict = {}
    users_trajs_labels_dict = {}
    users_trajs_lens_dict = {}
    users_trajs_lastpoi_group_dict = {}
    for userID, sessions in sessions_dict.items():
        labels_sessions=label_dict[userID]
        for id,session in enumerate(sessions):
            key_str = f"{userID}_{id}"
            users_trajs_labels_dict[key_str]= labels_sessions[id]
            users_trajs_dict[key_str] = [userID,session]
            users_trajs_lastpoi_group_dict[key_str] = session[-1]
            users_trajs_lens_dict[key_str] = len(session)
    return users_trajs_dict, users_trajs_labels_dict , users_trajs_lens_dict,users_trajs_lastpoi_group_dict



def get_traj_lasttime(sessions_dict):
    """Get each user's complete trajectory from her sessions"""
    traj_lasttime_dict = {}
   
    for userID, sessions in sessions_dict.items():
        for id,session in enumerate(sessions):
            key_str = f"{userID}_{id}"
            traj_lasttime_dict[key_str]= session[-1]
            

    return traj_lasttime_dict
    


def get_user_reverse_traj(users_trajs_dict):
    """Get each user's reversed trajectory according to her complete trajectory"""
    users_rev_trajs_dict = {}
    for userID, traj in users_trajs_dict.items():
        rev_traj = traj[::-1]
        users_rev_trajs_dict[userID] = rev_traj

    return users_rev_trajs_dict


def gen_poi_geo_adj(num_pois, pois_coos_dict, distance_threshold):
    """Generate geogrpahical adjacency matrix with pois_coos_dict and distance_threshold"""
    # poi_geo_adj = np.zeros(shape=(num_pois, num_pois))
    poi_geo_adj = sp.lil_matrix((num_pois, num_pois)) 
    
    for poi1 in tqdm(range(num_pois + 1), desc="gen poi geo adj"):
        if poi1 not in pois_coos_dict:
            continue 
        lat1, lon1,city_tag,main_category1 = pois_coos_dict[poi1][0] 
        for poi2 in range(poi1, num_pois+1):
            if poi2 not in pois_coos_dict:
                continue
            lat2, lon2,city_tag,main_category2 = pois_coos_dict[poi2][0]
            hav_dist = haversine_distance(lon1, lat1, lon2, lat2)
            if hav_dist <= distance_threshold:
                poi_geo_adj[poi1, poi2] = 1
                poi_geo_adj[poi2, poi1] = 1

    # transform np.ndarray to csr_matrix
    poi_geo_adj = sp.csr_matrix(poi_geo_adj)
    return poi_geo_adj


def process_users_seqs(users_seqs_dict, padding_idx, max_seq_len):
    processed_seqs_dict = {}
    reverse_seqs_dict = {}
    for key, seq in users_seqs_dict.items():
        if len(seq) >= max_seq_len:
            temp_seq = seq[-max_seq_len:]
            temp_rev_seq = temp_seq[::-1]
        else:
            temp_seq = seq + [padding_idx] * (max_seq_len - len(seq))
            temp_rev_seq = seq[::-1] + [padding_idx] * (max_seq_len - len(seq))
        processed_seqs_dict[key] = temp_seq
        reverse_seqs_dict[key] = temp_rev_seq

    return processed_seqs_dict, reverse_seqs_dict


def reverse_users_seqs(processed_users_seqs_dict, padding_idx, max_seq_len):
    reversed_users_seqs_dict = {}
    for key, seq in processed_users_seqs_dict.items():
        for idx in range(len(seq)):
            if seq[idx] == padding_idx:
                actual_seq = seq[:idx]
                reversed_users_seqs_dict[key] = actual_seq[::-1] + [padding_idx] * (max_seq_len - idx)
                break

    return reversed_users_seqs_dict


def gen_users_seqs_masks(users_seqs_dict, padding_idx):
    users_seqs_masks_dict = {}
    for key, seq in users_seqs_dict.items():
        temp_seq = []
        for poi in seq:
            if poi != padding_idx:
                temp_seq.append(1)
            else:
                temp_seq.append(0)
        users_seqs_masks_dict[key] = temp_seq

    return users_seqs_masks_dict


def haversine_distance(lon1, lat1, lon2, lat2):
    """Haversine distance"""
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371

    return c * r


def euclidean_distance(lon1, lat1, lon2, lat2):
    """Euclidean distance"""

    return np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)


def gen_geo_seqs_adjs_dict(users_seqs_dict, pois_coos_dict, max_seq_len, padding_idx, eta=1, distance_threshold=2.5, distance_type="haversine"):
    """Generate geographical sequential adjacency dictionary"""
    geo_adjs_dict = {}
    for key, seq in users_seqs_dict.items():
        geo_adj = np.zeros(shape=(max_seq_len, max_seq_len))
        actual_seq = []
        for item in seq:
            if item != padding_idx:
                actual_seq.append(item)
        actual_seq_len = len(actual_seq)
        for i in range(actual_seq_len):
            for j in range(i + 1, actual_seq_len):
                l1 = actual_seq[i]
                l2 = actual_seq[j]
                lat1, lon1,city_tag = pois_coos_dict[l1]
                lat2, lon2 ,city_tag= pois_coos_dict[l2]
                if distance_type == "haversine":
                    dist = haversine_distance(lon1, lat1, lon2, lat2)
                elif distance_type == "euclidean":
                    dist = euclidean_distance(lon1, lat1, lon2, lat2)
                if 0 < dist <= distance_threshold:
                    geo_influence = np.exp(-eta * (dist ** 2))
                    geo_adj[i, j] = geo_influence
                    geo_adj[j, i] = geo_influence
        geo_adjs_dict[key] = geo_adj

    return geo_adjs_dict


def create_user_poi_adj(users_seqs_dict, num_users, num_pois):
    """Create user-POI interaction matrix"""
    R = sp.dok_matrix((num_users, num_pois), dtype=np.float)
    for userID, seq in users_seqs_dict.items():
        for itemID in seq:
            itemID = itemID - num_users
            R[userID, itemID] = 1

    return R, R.T


def gen_sparse_interaction_matrix(users_seqs_dict, num_users, num_pois):
    """Generate sparse user-POI adjacent matrix"""
    R, R_T = create_user_poi_adj(users_seqs_dict, num_users, num_pois)
    A = sp.dok_matrix((num_users + num_pois, num_users + num_pois), dtype=float)
    A[:num_users, num_users:] = R
    A[num_users:, :num_users] = R_T
    A_sparse = A.tocsr()

    return A_sparse


def normalized_adj(adj, is_symmetric=True):
    """Normalize adjacent matrix for GCN"""
    if is_symmetric:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1/2).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj * d_mat_inv
    else:
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum + 1e-8, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv * adj

    return norm_adj


def normalized_adj_tensor(adj_tensor):
    """Normalized adjacent tensor"""
    # Compute the degree matrix
    degree_tensor = torch.diag(torch.sum(adj_tensor, dim=1))

    # inverse degree
    inverse_degree_tensor = torch.inverse(degree_tensor)

    # normalized adjacency
    norm_adj = torch.matmul(inverse_degree_tensor, adj_tensor)

    # convert the normalized adjacency matrix to a sparse tensor
    sparse_norm_adj = torch.sparse.FloatTensor(norm_adj)

    return sparse_norm_adj


def gen_local_graph(adj):
    """Add self loop"""
    G = normalized_adj(adj + sp.eye(adj.shape[0]))

    return G


def gen_sparse_H(sessions_dict, num_pois, num_sessions, start_poiID):
    """Generate sparse incidence matrix for hypergraph"""
    H = np.zeros(shape=(num_pois, num_sessions))
    sess_idx = 0
    for key, sessions in sessions_dict.items():
        for session in sessions:
            for poiID in session:
                new_poiID = poiID - start_poiID
                H[new_poiID, sess_idx] = 1
            sess_idx += 1
    assert sess_idx == num_sessions
    H = sp.csr_matrix(H)

    return H


def gen_sparse_H_pois_session(sessions_dict, num_pois, num_sessions):
    H = np.zeros(shape=(num_pois, num_sessions))
    for sess_idx, session in sessions_dict.items():
        for poi in session:
            H[poi, sess_idx] = 1
    H = sp.csr_matrix(H)

    return H


def gen_sparse_H_user(sessions_dict, num_pois,  num_users,user_label_dict=None,group_num=1):
    """Generate sparse incidence matrix for hypergraph"""
    # H = np.zeros(shape=(num_pois, num_users))
    # H=sp.lil_matrix((num_pois,num_users))
    
    if group_num > 1:
        H_groups={}
        for s in range(group_num):
            H_groups[s]=sp.lil_matrix((num_pois,num_users))
    else:
        H_group=sp.lil_matrix((num_pois,num_users))
    
    for userID, sessions in tqdm(sessions_dict.items(), desc="gen_sparse_H_poi_user"):
        seq = []
        for session in sessions:
            seq.extend(session)
            
        if group_num > 1:
            for poi in seq:
                H_groups[user_label_dict[str(userID)]][poi, int(userID)]+=1
        else:
            for poi in seq:
                H_group[poi, int(userID)]+=1
    
    
    if group_num > 1:
        for group,H in H_groups.items():
            H_groups[group]= sp.csr_matrix(H)
        return H_groups
    else:
        H_group = sp.csr_matrix(H_group)
        return H_group
    

def gen_sparse_H_time(sessions_dict, time_category, num_users,time_group_dict=None,group_num=1):
    """Generate sparse incidence matrix for hypergraph"""
    # H = np.zeros(shape=(num_pois, num_users))
    # H=sp.lil_matrix((num_pois,num_users))
    
    if group_num > 1:
        H_groups={}
        for s in range(group_num):
            H_groups[s]=sp.lil_matrix((time_category,num_users))
    else:
        H_group=sp.lil_matrix((time_category,num_users))
    
    
    for userID, sessions in tqdm(sessions_dict.items(), desc="gen_sparse_H_time_user"): 
        seq = [] 
        for session in sessions:
            seq.extend(session)
            
        if group_num > 1:
            for time_c in seq:
                H_groups[time_group_dict[time_c]][time_c, int(userID)]+=1
        else:
            for time_c in seq:
                H_group[time_c, int(userID)]+=1
    
    if group_num > 1:
        for group,H in H_groups.items():
            H_groups[group]= sp.csr_matrix(H)
        return H_groups
    else:
        H_group=sp.csr_matrix(H_group)
        return H_group
    
def gen_sparse_H_time_poi(sessions_dict, time_category, num_poi,group_dict=None,group_num=1):
    """Generate sparse incidence matrix for hypergraph"""
    # H = np.zeros(shape=(num_pois, num_users))
    # H=sp.lil_matrix((num_pois,num_users))
    
    if group_num > 1:
        H_groups={}
        for s in range(group_num):
            H_groups[s]=sp.lil_matrix((time_category,num_poi))
    else:
        H_group=sp.lil_matrix((time_category,num_poi))
    
    
    for userID, sessions in tqdm(sessions_dict.items(), desc="gen_sparse_H_Time_POI"): 
        seq = [] 
        for session in sessions:
            seq.extend(session)
            
        if group_num > 1:
            for (poi,time_c)  in seq:
                H_groups[group_dict[time_c]][time_c, poi]+=1
        else:
            for (poi,time_c) in seq:
                H_group[time_c, poi]+=1
    
    if group_num > 1:
        for group,H in H_groups.items():
            H_groups[group]= sp.csr_matrix(H)
        return H_groups
    else:
        H_group=sp.csr_matrix(H_group)
        return H_group
        
        
    


def gen_sparse_directed_H_poi_(users_trajs_dict, num_pois):
    """
    Generate directed poi-poi incidence matrix for hypergraph
    Rows: source POIs
    Columns: target POIs
    """
    # H = np.zeros(shape=(num_pois, num_pois))
    H= sp.lil_matrix((num_pois,num_pois))
    for userID, traj in tqdm(users_trajs_dict.items(),desc="gen_sparse_directed_H_poi"):
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                H[src_poi, tar_poi] = 1
    H = sp.csr_matrix(H)

    return H



def gen_sparse_directed_H_poi(users_trajs_dict, num_pois):
    """
    Generate directed poi-poi incidence matrix for hypergraph
    Rows: source POIs
    Columns: target POIs
    """
    # H = np.zeros(shape=(num_pois, num_pois))
    H= sp.lil_matrix((num_pois,num_pois))
    for traj_id, user_trajs in tqdm(users_trajs_dict.items(),desc="gen_sparse_directed_H_poi"):
        traj=user_trajs[1]
        
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                H[src_poi, tar_poi] = 1
    H = sp.csr_matrix(H)

    return H

def calculate_laplacian_matrix(adj_mat):
    """Calculate Laplacian matrix using DGL for better performance"""
    
    # Create DGL graph from scipy sparse matrix
    g = dgl.from_scipy(adj_mat)
    
    # Get number of nodes
    n_vertex = g.number_of_nodes()
    adj_mat_coo = adj_mat.tocoo()
    indices = torch.LongTensor([adj_mat_coo.row, adj_mat_coo.col])
    values = torch.FloatTensor(adj_mat_coo.data)
    adj_tensor = torch.sparse_coo_tensor(indices, values, size=adj_mat.shape).to_dense()
    deg_tensor = torch.sum(adj_tensor, dim=1)
    deg_mat = torch.diag(deg_tensor)
    id_mat = torch.eye(n_vertex, dtype=adj_tensor.dtype, device=adj_tensor.device)
    wid_deg_mat = deg_mat + id_mat
    wid_adj_mat = adj_tensor + id_mat
    deg_inv = torch.diag(1.0 / (torch.diag(wid_deg_mat) + 1e-8))
    hat_rw_normd_lap_mat = torch.mm(deg_inv, wid_adj_mat)
    
    return hat_rw_normd_lap_mat
   

def gen_attn_A(users_trajs_dict, num_pois):
    """
    Generate directed poi-poi incidence matrix for hypergraph
    Rows: source POIs
    Columns: target POIs
    """
    # H = np.zeros(shape=(num_pois, num_pois))
    H= sp.lil_matrix((num_pois,num_pois))
    
    for traj_id, user_trajs in tqdm(users_trajs_dict.items(),desc="gen_sparse_attn_A"):
        traj=user_trajs[1]
        
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                H[src_poi, tar_poi] += 1
    H = sp.csr_matrix(H)
    # H=transform_csr_matrix_to_tensor(H)
    H=calculate_laplacian_matrix(H)

    return H

def gen_attn_X(users_trajs_dict,pois_coos_dict,num_pois):
    """
    Generate X matrix for attn model
    Rows: checkin_num
    Columns: longitude
    """
    X=torch.zeros((num_pois,2))
    
    for traj_id, user_trajs in tqdm(users_trajs_dict.items(),desc="gen_attn_X"):
        traj=user_trajs[1]
        for poi_idx in traj:
            X[poi_idx,0]+=1
            if X[poi_idx,1] ==0:
                X[poi_idx,1] = pois_coos_dict[poi_idx][0][1]
    return X


def gen_HG_from_sparse_H(H, conv="sym"):
    """Generate hypergraph with sparse incidence matrix"""
    n_edge = H.shape[1]
    W = sp.eye(n_edge)

    HW = H.dot(W)
    DV = sp.csr_matrix(HW.sum(axis=1)).astype(float)
    DE = sp.csr_matrix(H.sum(axis=0)).astype(float)
    invDE1 = DE.power(-1)
    invDE1_ = sp.diags(invDE1.toarray()[0])
    HT = H.T

    if conv == "sym":
        invDV2 = DV.power(n=-1 / 2)
        invDV2_ = sp.diags(invDV2.toarray()[:, 0])
        HG = invDV2_ * H * W * invDE1_ * HT * invDV2_
    elif conv == "asym":
        invDV1 = DV.power(-1)
        invDV1_ = sp.diags(invDV1.toarray()[:, 0])
        HG = invDV1_ * H * W * invDE1_ * HT

    return HG


def get_hyper_deg(incidence_matrix):
    '''
    # incidence_matrix = [num_nodes, num_hyperedges]
    hyper_deg = np.array(incidence_matrix.sum(axis=axis)).squeeze()
    hyper_deg[hyper_deg == 0.] = 1
    hyper_deg = sp.diags(1.0 / hyper_deg)
    '''
    rowsum = np.array(incidence_matrix.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    return d_mat_inv


def transform_csr_matrix_to_tensor(csr_matrix):
    """Transform csr matrix to tensor"""
    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    sp_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return sp_tensor


def get_poi_session_freq(num_pois, num_sessions, sessions_dict):
    """Get frequency occurrence of pois in sessions"""
    poi_sess_freq_matrix = np.zeros(shape=(num_pois, num_sessions))

    # traverse
    sess_idx = 0
    for userID, sessions in sessions_dict.items():
        for session in sessions:
            for poiID in session:
                poi_sess_freq_matrix[poiID, sess_idx] += 1
            sess_idx += 1

    # transform to csr_matrix
    poi_sess_freq_matrix = sp.csr_matrix(poi_sess_freq_matrix)

    return poi_sess_freq_matrix


def get_all_sessions(sessions_dict):
    """Get all sessions in the dataset"""
    all_sessions = []

    for userID, sessions in sessions_dict.items():
        for session in sessions:
            all_sessions.append(torch.tensor(session))

    return all_sessions


def get_all_users_seqs(users_trajs_dict):
    """Get all users' sequences"""
    all_seqs = []
    for userID, traj in users_trajs_dict.items():
        all_seqs.append(torch.tensor(traj))

    return all_seqs


def sparse_adj_tensor_drop_edge(sp_adj, keep_rate):
    """Drop edge on sparse adjacency tensor"""
    if keep_rate == 1.0:
        return sp_adj

    vals = sp_adj._values()
    idxs = sp_adj._indices()
    edgeNum = vals.size()
    mask = ((torch.rand(edgeNum) + keep_rate).floor()).type(torch.bool)
    newVals = vals[mask] / keep_rate
    newIdxs = idxs[:, mask]

    return torch.sparse.FloatTensor(newIdxs, newVals, sp_adj.shape)


def csr_matrix_drop_edge(csr_adj_matrix, keep_rate):
    """Drop edge on scipy.sparse.csr_matrix"""
    if keep_rate == 1.0:
        return csr_adj_matrix

    coo = csr_adj_matrix.tocoo()
    row = coo.row
    col = coo.col
    edgeNum = row.shape[0]

    # generate edge mask
    mask = np.floor(np.random.rand(edgeNum) + keep_rate).astype(np.bool_)

    # get new values and indices
    new_row = row[mask]
    new_col = col[mask]
    new_edgeNum = new_row.shape[0]
    new_values = np.ones(new_edgeNum, dtype=np.float)

    drop_adj_matrix = sp.csr_matrix((new_values, (new_row, new_col)), shape=coo.shape)

    return drop_adj_matrix



def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss



def pareto_check(table_results, eps_pareto=0):

    #table_results dimension: objectives x iterations
    pareto = np.zeros([table_results.shape[1]])
    if table_results.shape[1] == 1:
        pareto[0] = True
        # print('one')
    else:
        for i in np.arange(table_results.shape[1]):
            dif_risks = table_results[:, i][:, np.newaxis] - table_results  #(dim: objectives x iterations)
            dif_risks = dif_risks < eps_pareto  #evaluates if item iteration i dominates some coordinate, 1 indicates dominance
            dif_risks[:,i] = 1 #we would like to have a row full of ones
            dif_risks = np.sum(dif_risks,axis = 0) #sum on objectives
            pareto[i] = (np.prod(dif_risks) > 0) #prod of iterations (I cannot have an iteration that = 0 when sum all objectives differences

    if np.sum(pareto) == 0: #if all vectors are the same basically...
        ix_p = np.argmin(np.sum(table_results,axis=0))
        pareto[ix_p] = 1
    return pareto



def extract_weight_method_parameters_from_args(args):
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=1,
                optim_niter=20,
            ),
            stl=dict(main_task=0),
            cagrad=dict(c=0.4),
            dwa=dict(temp=2.0),
        )
    )
    return weight_methods_parameters

def load_graph_node_features(data_filename, pois_coos_filename,feature1='checkin_cnt', feature2='latitude',
                             feature3='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    
    checkin_data = load_list_with_pkl(data_filename)  # data = [sessions_dict, labels_dict]
    pois_coos_dict = load_dict_from_pkl(pois_coos_filename)
   
    ##读取文件
    rlt_df = df[[feature1, feature2, feature3]]
    X = rlt_df.to_numpy()

    return X


def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A