import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.utils.rnn import pad_sequence





class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x



def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)



class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
       

    def forward(self, x):
        x = self.l1(x)
        return x
    
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class TransformerModel(nn.Module):
    def __init__(self, num_poi,  embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)

        return out_poi, out_time

class MultiViewHyperConvLayer(nn.Module):
    """
    Multi-view Hypergraph Convolutional Layer
    """

    def __init__(self, emb_dim, device):
        super(MultiViewHyperConvLayer, self).__init__()
        self.fc_fusion = nn.Linear(2 * emb_dim, emb_dim, device=device)
        self.dropout = nn.Dropout(0.5)
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, pois_embs, HG_up, HG_pu):
        msg_poi_agg = torch.sparse.mm(HG_up, pois_embs)  # [U, d]
        propag_pois_embs = torch.sparse.mm(HG_pu, msg_poi_agg)  # [L, d]
        return propag_pois_embs


class DirectedHyperConvLayer(nn.Module):
    """Directed hypergraph convolutional layer"""

    def __init__(self):
        super(DirectedHyperConvLayer, self).__init__()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        msg_tar = torch.sparse.mm(HG_poi_tar, pois_embs)
        msg_src = torch.sparse.mm(HG_poi_src, msg_tar)

        return msg_src


class MultiViewHyperConvNetwork(nn.Module):
    """
    Multi-view Hypergraph Convolutional Network
    """

    def __init__(self, num_layers, emb_dim, dropout, device):
        super(MultiViewHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.mv_hconv_layer = MultiViewHyperConvLayer(emb_dim, device)
        self.dropout = dropout

    def forward(self, pois_embs, HG_up, HG_pu):
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.mv_hconv_layer(pois_embs, HG_up, HG_pu)  # [L, d]
            # add residual connection to alleviate over-smoothing issue
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)  # [L, d]

        return final_pois_embs


class DirectedHyperConvNetwork(nn.Module):
    def __init__(self, num_layers, device, dropout=0.3):
        super(DirectedHyperConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.device = device
        self.dropout = dropout
        self.di_hconv_layer = DirectedHyperConvLayer()

    def forward(self, pois_embs, HG_poi_src, HG_poi_tar):
        final_pois_embs = [pois_embs]
        for layer_idx in range(self.num_layers):
            pois_embs = self.di_hconv_layer(pois_embs, HG_poi_src, HG_poi_tar)
            # add residual connection
            pois_embs = pois_embs + final_pois_embs[-1]
            pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        final_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)  # [L, d]

        return final_pois_embs


class GeoConvNetwork(nn.Module):
    def __init__(self, num_layers, dropout):
        super(GeoConvNetwork, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, pois_embs, geo_graph):
        final_pois_embs = [pois_embs]
        for _ in range(self.num_layers):
            # pois_embs = geo_graph @ pois_embs
            pois_embs = torch.sparse.mm(geo_graph, pois_embs)
            pois_embs = pois_embs + final_pois_embs[-1]
            # pois_embs = F.dropout(pois_embs, self.dropout)
            final_pois_embs.append(pois_embs)
        output_pois_embs = torch.mean(torch.stack(final_pois_embs), dim=0)  # [L, d]

        return output_pois_embs


class MSAHG(nn.Module):
    def __init__(self, num_users, num_pois, args, device):
        super(MSAHG, self).__init__()

        # definition
        # self.user_label_dict = user_label_dict
        self.num_pois = num_pois
        self.num_users=num_users
        self.args = args
        self.device = device
        self.emb_dim = args.emb_dim
        self.ssl_temp = args.temperature
        self.specific_parameters_set=set()
        self.task_specific_parameters={}
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(num_pois + 1, self.emb_dim, padding_idx=-1)
        


        # network
        self.mv_hconv_network = MultiViewHyperConvNetwork(args.num_mv_layers, args.emb_dim,  args.dropout, device)
        self.mv_time_user_network = MultiViewHyperConvNetwork(args.num_mv_layers, args.emb_dim,  args.dropout, device)
        self.mv_time_poi_network = MultiViewHyperConvNetwork(args.num_mv_layers, args.emb_dim,  args.dropout, device)
        self.geo_conv_network = GeoConvNetwork(args.num_geo_layers, args.dropout)
        self.di_hconv_network = DirectedHyperConvNetwork(args.num_di_layers, device, args.dropout)

        # gate for adaptive fusion with users embeddings
        self.user_trans_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.user_hyper_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.user_gcn_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.user_time_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
    
     

        # gating before disentangled learning
        self.w_gate_geo = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_geo = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_seq = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_seq = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        self.w_gate_col = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
       
        self.w_gate_time  = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_time = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
       
        self.w_gate_t2u  = nn.Parameter(torch.FloatTensor(args.emb_dim, args.emb_dim))
        self.b_gate_t2u = nn.Parameter(torch.FloatTensor(1, args.emb_dim))
        
        
        #poi
        nn.init.xavier_normal_(self.w_gate_geo.data)
        nn.init.xavier_normal_(self.b_gate_geo.data)
        nn.init.xavier_normal_(self.w_gate_seq.data)
        nn.init.xavier_normal_(self.b_gate_seq.data)
        nn.init.xavier_normal_(self.w_gate_col.data)
        nn.init.xavier_normal_(self.b_gate_col.data)
        nn.init.xavier_normal_(self.w_gate_time.data)
        nn.init.xavier_normal_(self.b_gate_time.data)
        #user
        nn.init.xavier_normal_(self.w_gate_t2u.data)
        nn.init.xavier_normal_(self.b_gate_t2u.data)
        
        

        # dropout
        self.dropout = nn.Dropout(args.dropout)

    @staticmethod
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]

        return corrupted_embedding

    def cal_poi_loss_infonce(self, emb1, emb2):
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        # neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        neg_score=self.cal_neg_sample_loss(emb1,emb2)
        
        loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
        loss /= pos_score.shape[0]
        torch.cuda.empty_cache()
        return loss
    
    
    def cal_neg_sample_loss(self, emb1, emb2, num_samples=1000):
        num_negatives = emb2.size(0)

        sample_indices = torch.randint(0, num_negatives, (num_samples,), device=emb1.device)
        emb2_sampled = emb2[sample_indices]
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2_sampled.T) / self.ssl_temp), axis=1)
        neg_score = neg_score * (num_negatives / num_samples)
        
        return neg_score
    
    
    def cal_loss_infonce(self, emb1, emb2):
        pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.ssl_temp)
        neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.ssl_temp), axis=1)
        loss = -torch.log(pos_score / (neg_score + 1e-8) + 1e-8)
        return loss

    def cal_loss_cl_pois(self, hg_pois_embs, geo_pois_embs, trans_pois_embs,time_pois_embs):

        # normalization
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)
        norm_time_pois_embs = F.normalize(time_pois_embs, p=2, dim=1)

        # calculate loss
        loss_cl_pois = 0.0
        loss_cl_pois += self.cal_poi_loss_infonce(norm_hg_pois_embs, norm_geo_pois_embs)
        loss_cl_pois += self.cal_poi_loss_infonce(norm_hg_pois_embs, norm_trans_pois_embs)
        loss_cl_pois += self.cal_poi_loss_infonce(norm_hg_pois_embs, norm_time_pois_embs)
        
        loss_cl_pois += self.cal_poi_loss_infonce(norm_geo_pois_embs, norm_trans_pois_embs)
        loss_cl_pois += self.cal_poi_loss_infonce(norm_geo_pois_embs, norm_time_pois_embs)
        loss_cl_pois += self.cal_poi_loss_infonce(norm_trans_pois_embs, norm_time_pois_embs)

        return loss_cl_pois

    def cal_loss_cl_users(self, hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs,batch_time_user_embs,device):
        # normalization
        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)
        norm_time_batch_users_embs = F.normalize(batch_time_user_embs, p=2, dim=1)
        loss_cl_users= torch.zeros(norm_hg_batch_users_embs.shape[0]).to(device)
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_geo_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_trans_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_hg_batch_users_embs, norm_time_batch_users_embs)
        
        loss_cl_users += self.cal_loss_infonce(norm_geo_batch_users_embs, norm_trans_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_geo_batch_users_embs, norm_time_batch_users_embs)
        loss_cl_users += self.cal_loss_infonce(norm_trans_batch_users_embs, norm_time_batch_users_embs)

        return loss_cl_users
    
    
    
    def add_task_specific_params(self,task_groups):
        named_params = dict(self.named_parameters())
        for param_name in task_groups.keys():
            self.specific_parameters_set.add(param_name)
            for group_id in task_groups[param_name].keys():
                
                original_param = named_params[param_name]
                copy_param = nn.Parameter(original_param.data.clone(), requires_grad=True)
                safe_param_name = f'task_{param_name.replace(".", "_")}'
                self.register_parameter(safe_param_name, copy_param)
                
                for task_idx in task_groups[param_name][group_id]:
                    if task_idx not in self.task_specific_parameters:
                        self.task_specific_parameters[task_idx] = {}
                    self.task_specific_parameters[task_idx][param_name] = copy_param

    def update_weights(self, task_specific_params):
        
        for name in task_specific_params.keys():
            if '.' in name:
                module_name, param_name = name.rsplit('.', 1)
                module = self
                for part in module_name.split('.'):
                    module = getattr(module, part)
                setattr(module, param_name, task_specific_params[name])
            else:
                setattr(self, name, task_specific_params[name])
        
          
       


    def forward(self, dataset, batch, task_idx=None):
        
        if task_idx is not None and len(self.task_specific_parameters)!=0 :
            self.update_weights(self.task_specific_parameters[task_idx])
       
            

        geo_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_geo) + self.b_gate_geo))
        seq_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_seq) + self.b_gate_seq))
        col_gate_pois_embs = torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_col) + self.b_gate_col))
        
        time_gate_pois_embs= torch.multiply(self.poi_embedding.weight[:-1],
                                            torch.sigmoid(torch.matmul(self.poi_embedding.weight[:-1],
                                                                       self.w_gate_time) + self.b_gate_time))
        
        
        time_gate_users_embs= torch.multiply(self.user_embedding.weight,
                                            torch.sigmoid(torch.matmul(self.user_embedding.weight,
                                                                       self.w_gate_t2u) + self.b_gate_t2u))
       
       
      
        if dataset.divide_group=='User':
            s=batch["group"][0].item()
            hg_pois_embs=self.mv_hconv_network(col_gate_pois_embs, dataset.HG_up[s], dataset.HG_pu[s])
            # hypergraph structure aware users embeddings
            hg_structural_users_embs = torch.sparse.mm(dataset.HG_up[s], hg_pois_embs)  # [U, d]
        else:
            hg_pois_embs=self.mv_hconv_network(col_gate_pois_embs, dataset.HG_up, dataset.HG_pu)
            # hypergraph structure aware users embeddings
            hg_structural_users_embs = torch.sparse.mm(dataset.HG_up, hg_pois_embs)  # [U, d]
        hg_batch_users_embs=(hg_structural_users_embs[batch["user_idx"]])   # [BS, d]
        
        
        if dataset.divide_group=='Time':
            s=batch["group"][0].item()
            time_pois_embs=self.mv_time_poi_network(time_gate_pois_embs,  dataset.HG_tp[s], dataset.HG_pt[s]) # [L, d]
        else:
            time_pois_embs=self.mv_time_poi_network(time_gate_pois_embs,  dataset.HG_tp, dataset.HG_pt) # [L, d]
       
        
        if dataset.divide_group=='Time':
            time_user_embs=self.mv_time_user_network(time_gate_users_embs, dataset.HG_tu[s], dataset.HG_ut[s]) # [U, d]
        else:
            time_user_embs=self.mv_time_user_network(time_gate_users_embs,  dataset.HG_tu, dataset.HG_ut) # [U, d] 
        batch_time_user_embs=(time_user_embs[batch["user_idx"]])
       
    
        # poi-poi geographical graph convolutional network
        geo_pois_embs = self.geo_conv_network(geo_gate_pois_embs, dataset.poi_geo_graph)  # [L, d]
        
        if dataset.divide_group=='User':
            # geo-aware user embeddings
            geo_structural_users_embs=torch.sparse.mm(dataset.HG_up[s], geo_pois_embs)
        else:
            # geo-aware user embeddings
            geo_structural_users_embs=torch.sparse.mm(dataset.HG_up, geo_pois_embs)
       
        geo_batch_users_embs = geo_structural_users_embs[batch["user_idx"]] # [BS, d]


        # poi-poi directed hypergraph
        trans_pois_embs = self.di_hconv_network(seq_gate_pois_embs, dataset.HG_poi_src, dataset.HG_poi_tar)

        if dataset.divide_group=='User':
            # transition-aware user embeddings
            trans_structural_users_embs=torch.sparse.mm(dataset.HG_up[s], trans_pois_embs)
        else:
             # transition-aware user embeddings
            trans_structural_users_embs=torch.sparse.mm(dataset.HG_up, trans_pois_embs)
        trans_batch_users_embs = trans_structural_users_embs[batch["user_idx"]]  # [BS, d]

        loss_cl_poi = self.cal_loss_cl_pois(hg_pois_embs, geo_pois_embs, trans_pois_embs,time_pois_embs)
        loss_cl_user = self.cal_loss_cl_users(hg_batch_users_embs, geo_batch_users_embs, trans_batch_users_embs,batch_time_user_embs,self.device)
        
        norm_hg_batch_users_embs = F.normalize(hg_batch_users_embs, p=2, dim=1)
        norm_geo_batch_users_embs = F.normalize(geo_batch_users_embs, p=2, dim=1)
        norm_trans_batch_users_embs = F.normalize(trans_batch_users_embs, p=2, dim=1)
        norm_time_batch_users_embs = F.normalize(batch_time_user_embs, p=2, dim=1)
            
        # adaptive fusion for user embeddings
        hyper_coef = self.user_hyper_gate(norm_hg_batch_users_embs)
        geo_coef = self.user_gcn_gate(norm_geo_batch_users_embs)
        trans_coef = self.user_trans_gate(norm_trans_batch_users_embs)
        time_coef = self.user_time_gate(norm_time_batch_users_embs)
        
        # normalization
        norm_geo_pois_embs = F.normalize(geo_pois_embs, p=2, dim=1)
        norm_trans_pois_embs = F.normalize(trans_pois_embs, p=2, dim=1)
        norm_hg_pois_embs = F.normalize(hg_pois_embs, p=2, dim=1)
        norm_time_pois_embs = F.normalize(time_pois_embs, p=2, dim=1)        
        
        fusion_batch_users_embs=hyper_coef*norm_hg_batch_users_embs+ geo_coef * norm_geo_batch_users_embs +  trans_coef* norm_trans_batch_users_embs + time_coef* norm_time_batch_users_embs
        fusion_pois_embs = norm_hg_pois_embs  + norm_geo_pois_embs + norm_trans_pois_embs + norm_time_pois_embs
        
        user_POI_perdict= fusion_batch_users_embs @ fusion_pois_embs.T
        
        return user_POI_perdict,loss_cl_poi,loss_cl_user


