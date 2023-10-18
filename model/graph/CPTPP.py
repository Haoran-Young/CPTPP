##########################
# This code take SGL, implemented by Coder-Yu on Github, as the backbone.
##########################


from turtle import forward
import torch
torch.manual_seed(12345)
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor
from sklearn.decomposition import NMF
import numpy as np

# Paper: self-supervised graph learning for recommendation. SIGIR'21


class CPTPP(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(CPTPP, self).__init__(conf, training_set, test_set)

        args = OptionConf(self.config['CPTPP'])
        self.cl_rate = float(args['-lambda'])
        aug_type = self.aug_type = int(args['-augtype'])
        drop_rate = float(args['-droprate'])
        self.n_layers = int(args['-n_layer'])
        temp = float(args['-temp'])
        self.inputs_type = int(args['-inputs_type'])
        prompt_size = int(args['-prompt_size'])

        self.model = SGL_Encoder(self.data, self.emb_size, drop_rate, self.n_layers, temp, aug_type)
        self.prompts_generator = Prompts_Generator(self.emb_size, prompt_size).cuda()
        self.fusion_mlp = Fusion_MLP(self.emb_size, prompt_size).cuda()

        if self.inputs_type == 0:
            self.interaction_mat = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda()
        if self.inputs_type == 2:
            # small dataset
            # self.adj_sparse = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.ui_adj)
            # self.ui_high_order = torch.sparse.mm(self.adj_sparse, self.adj_sparse.to_dense()).cuda()

            # big dataset Ciao
            self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda()

    def _pre_train(self):
        pre_trained_model = self.model.cuda()
        optimizer = torch.optim.Adam(pre_trained_model.parameters(), lr=self.lRate)

        print('############## Pre-Training Phase ##############')
        for epoch in range(self.maxPreEpoch):
            dropped_adj1 = pre_trained_model.graph_reconstruction()
            dropped_adj2 = pre_trained_model.graph_reconstruction()

            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                cl_loss = pre_trained_model.cal_cl_loss([user_idx,pos_idx],dropped_adj1,dropped_adj2)
                batch_loss = cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                if epoch == self.maxEpoch-1:
                    batch_loss.backward(retain_graph=True)
                else:
                    batch_loss.backward()
                optimizer.step()
                if n % 100==0:
                    print('pre-training:', epoch + 1, 'batch', n, 'cl_loss', cl_loss.item())

    def _csr_to_pytorch_dense(self, csr):
        array = csr.toarray()
        dense = torch.Tensor(array)
        return dense.cuda()

    def _prompts_generation(self, item_emb, user_emb):
        if self.inputs_type == 0:
            inputs = self._historical_records(item_emb)
        # elif self.inputs_type == 1:
        #     inputs = self._adjacency_matrix_factorization()
        elif self.inputs_type == 2:
            inputs = self._high_order_u_relations(item_emb, user_emb)
        prompts = self.prompts_generator(inputs)
        return prompts

    def _historical_records(self, item_emb):
        inputs = torch.mm(self.interaction_mat, item_emb)
        return inputs

    # def _adjacency_matrix_factorization(self):
    #     adjacency_matrix = self.data.interaction_mat
    #     adjacency_matrix = adjacency_matrix.toarray()

    #     print('######### Adjacency Matrix Factorization #############')
    #     nmf = NMF(n_components=self.emb_size)
    #     user_profiles = nmf.fit_transform(adjacency_matrix)
    #     inputs = torch.Tensor(user_profiles).cuda()
    #     return inputs

    def _high_order_u_relations(self, item_emb, user_emb):
        # small dataset
        # emb = torch.cat((user_emb, item_emb), 0)
        # inputs = torch.sparse.mm(self.ui_high_order, emb)
        # inputs = inputs[:self.data.user_num, :]
        # return inputs

        # big dataset Ciao
        ego_embeddings = torch.cat((user_emb, item_emb), 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        inputs, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return inputs

    def _prompts_u_embeddings_fusion(self, prompts, user_emb):
        prompts_user_emb = torch.cat((prompts, user_emb), 1)
        prompted_user_emb = self.fusion_mlp(prompts_user_emb)
        return prompted_user_emb

    def train(self):
        self._pre_train()

        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        if self.inputs_type == 1:
            nmf = NMF(n_components=self.emb_size, max_iter=1000)
            self.user_profiles = torch.Tensor(nmf.fit_transform(self.data.interaction_mat.toarray())).cuda()

        print('############## Downstream Training Phase ##############')
        for epoch in range(self.maxEpoch):
            # dropped_adj1 = model.graph_reconstruction()
            # dropped_adj2 = model.graph_reconstruction()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_emb, item_emb = model()
                if self.inputs_type == 0 or self.inputs_type == 2:
                    prompts = self._prompts_generation(item_emb, user_emb)
                else:
                    prompts = self.prompts_generator(self.user_profiles)
                prompted_user_emb = self._prompts_u_embeddings_fusion(prompts, user_emb)

                user_idx, pos_idx, neg_idx = batch
                # rec_user_emb, rec_item_emb = model()
                rec_user_emb, rec_item_emb = prompted_user_emb, item_emb
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                # cl_loss = self.cl_rate * model.cal_cl_loss([user_idx,pos_idx],dropped_adj1,dropped_adj2)
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) #+ cl_loss
                # Backward and optimize
                
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if n % 100==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())#, 'cl_loss', cl_loss.item())
            with torch.no_grad():
                user_emb, self.item_emb = self.model()
                if self.inputs_type == 0 or self.inputs_type == 2:
                    prompts = self._prompts_generation(self.item_emb, user_emb)
                else:
                    prompts = self.prompts_generator(self.user_profiles)
                prompted_user_emb = self._prompts_u_embeddings_fusion(prompts, user_emb)
                self.user_emb = prompted_user_emb
            if epoch>=5:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

        #### save user embeddings
        # np_user_emb = self.user_emb.cpu().numpy()
        # np.save('./user_emb/cptpp-r-gowalla.npy', np_user_emb)

    def save(self):
        with torch.no_grad():
            best_user_emb, self.best_item_emb = self.model.forward()
            if self.inputs_type == 0 or self.inputs_type == 2:
                prompts = self._prompts_generation(self.best_item_emb, best_user_emb)
            else:
                prompts = self.prompts_generator(self.user_profiles)
            prompted_user_emb = self._prompts_u_embeddings_fusion(prompts, best_user_emb)
            self.best_user_emb = prompted_user_emb

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class Prompts_Generator(nn.Module):
    def __init__(self, emb_size, prompt_size):
        super(Prompts_Generator, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(emb_size, prompt_size), nn.Linear(prompt_size, prompt_size)])
        self.activation = nn.Tanh()
        #self.activation = nn.Sigmoid()

    def forward(self, inputs):
        prompts = inputs
        for i in range(len(self.layers)):
            prompts = self.layers[i](prompts)
            prompts = self.activation(prompts)
        
        return prompts


class Fusion_MLP(nn.Module):
    def __init__(self, emb_size, prompt_size):
        super(Fusion_MLP, self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(emb_size+prompt_size, emb_size), nn.Linear(emb_size, emb_size)])
        self.activation = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activation(x)
        
        return x


class SGL_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, temp, aug_type):
        super(SGL_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.temp = temp
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self, perturbed_adj=None):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj,list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        #return user_cl_loss + item_cl_loss
        return InfoNCE(view1,view2,self.temp)