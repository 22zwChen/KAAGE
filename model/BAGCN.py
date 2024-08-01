# BAGCN # A+B+C

agg = "gate2"
# gate2, gate1, gate, MLP, original

import torch
import torch.nn
from helper import *


class Gate(nn.Module):
    def __init__(self,
                 input1_size , # 200
                 input2_size , # 400 or 200
                 gate_activation=torch.sigmoid):
        super(Gate, self).__init__()
        self.gate_activation = gate_activation
        self.g = nn.Linear(input1_size + input2_size, input1_size)  
        self.g1 = nn.Linear(input1_size, input1_size, bias=False)
        self.g2 = nn.Linear(input2_size, input1_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(input1_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], x_lit.ndimension() - 1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1 - gate) * x_ent + gate * g_embedded

        return output
    

class BAGCN(torch.nn.Module):


    sub_ent_dim = 0  #
    obj_ent_dim = 1  #

    ent_dim = 0  #
    rel_dim = 0

    def __init__(self, in_channels, out_channels, activation= lambda x:x, dropout=None, residual=None, bias=None):
        super().__init__()

        """

        Parameters
        ----------
        in_channels
        out_channels
        num_heads
        concat
        activation
        dropout
        residual
        bias
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual

        self.w_ent_sub = torch.nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.w_ent_obj = torch.nn.Linear(self.in_channels, self.in_channels, bias=False)
        self.w_rel = torch.nn.Linear(self.in_channels, self.in_channels, bias=False)

        self.leakyReLU = nn.LeakyReLU(0.2)

        self.a = get_param((1, self.in_channels))
        # self.b = get_param((1, self.in_channels))

        self.kernel_in = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.kernel_out = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.kernel_rel = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.activation = activation

        if agg == "MLP": #MLP
            self.mlp_600_200 = nn.Sequential(  
                nn.Linear(600, 400),  # 输入维度是600  
                nn.ReLU(),  
                nn.Linear(400, 200)  # 输出维度应与in_ent_embed的维度相同，即200  
            ) 

            self.mlp_400_200 = nn.Sequential(  
                nn.Linear(400, 300),  # 输入维度是400  
                nn.ReLU(),  
                nn.Linear(300, 200)  # 输出维度应与in_ent_embed的维度相同，即200  
            ) 
        if agg == "gate":   # gate
            self.gate_ent = Gate(200,200)

        if agg == "gate1":   # gate
            self.gate1_ent = Gate(200,400)

        if agg == "gate2":   # gate
            self.gate2_ent = Gate(200,400)
            self.gate2_rel = Gate(200,200)
        # self.bn_ent = torch.nn.BatchNorm1d(out_channels)
        # self.bn_rel = torch.nn.BatchNorm1d(out_channels)

        self.residual_proj_ent = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.residual_proj_rel = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)

        if bias:
            self.bias_ent = get_param((1, self.out_channels))
            self.bias_rel = get_param((1, self.out_channels))



    def forward(self, ent_embed, rel_embed, edge_index, edge_type):


        num_edges = edge_index.size(1) // 2  # E
        num_ent = ent_embed.size(0)  # N
        num_rel = rel_embed.size(0) // 2
        
        # Step 1: Linear Projection + regularization
        # shape = (N, FIN)
        sub_ent_embed_proj = self.w_ent_sub(ent_embed)
        obj_ent_embed_proj = self.w_ent_obj(ent_embed)
        # shape = (R, FIN)
        rel_embed_proj = self.w_rel(rel_embed)

        # Step 2, 3
        in_index, out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        in_type, out_type = edge_type[:num_edges], edge_type[num_edges:]

        # shape = (N, FOUT) shape = (R, FOUT)
        in_ent_embed,  in_rel_embed= self.aggregate_neighbors(sub_ent_embed_proj, obj_ent_embed_proj, rel_embed_proj, in_index, in_type, ent_embed, num_ent, num_rel, mode='in')
        out_ent_embed, out_rel_embed = self.aggregate_neighbors(sub_ent_embed_proj, obj_ent_embed_proj, rel_embed_proj, out_index, out_type, ent_embed, num_ent, num_rel, mode='out')
        
        if agg == "original":    # original
            update_ent_embed = in_ent_embed + out_ent_embed

            update_rel_embed = torch.cat([in_rel_embed, out_rel_embed], dim=0)
            # in_rel_embed.shape = out_rel_embed.shape = torch.Size([number of relation, gcn_dim])
            # self.residual_proj_rel(rel_embed).shape = torch.Size([number of relation * 2, gcn_dim])

            # Step 4: Residual/skip connections, bias
            # shape = (N, FOUT)
            if self.residual: # True
                update_ent_embed = update_ent_embed + self.residual_proj_ent(ent_embed)

                update_rel_embed = update_rel_embed + self.residual_proj_rel(rel_embed)

        if agg == "MLP":    # MLP
            concat_ent_embed = torch.cat([in_ent_embed, out_ent_embed, self.residual_proj_ent(ent_embed)], dim=1)
            update_ent_embed = self.mlp_600_200(concat_ent_embed)

            update_rel_embed = torch.cat([in_rel_embed, out_rel_embed], dim=0)
            if self.residual: # True
                update_ent_embed = update_ent_embed + self.residual_proj_ent(ent_embed)

                update_rel_embed = update_rel_embed + self.residual_proj_rel(rel_embed)

        if agg == "gate":   # gate
            update_ent_embed = self.gate_ent(in_ent_embed,out_ent_embed)

            update_rel_embed = torch.cat([in_rel_embed, out_rel_embed], dim=0)

            if self.residual: # True
                update_ent_embed = update_ent_embed + self.residual_proj_ent(ent_embed)

                update_rel_embed = update_rel_embed + self.residual_proj_rel(rel_embed)

        if agg == "gate1":   # gate1           
            concat_ent_embed = torch.cat([in_ent_embed, out_ent_embed], dim=1)
            update_ent_embed = self.gate1_ent(self.residual_proj_ent(ent_embed),concat_ent_embed)

            update_rel_embed = torch.cat([in_rel_embed, out_rel_embed], dim=0)
            update_rel_embed = update_rel_embed + self.residual_proj_rel(rel_embed)
        
        if agg == "gate2":   # gate2
            concat_ent_embed = torch.cat([in_ent_embed, out_ent_embed], dim=1)
            update_ent_embed = self.gate2_ent(self.residual_proj_ent(ent_embed),concat_ent_embed)

            update_rel_embed = torch.cat([in_rel_embed, out_rel_embed], dim=0)
            update_rel_embed = self.gate2_rel(update_rel_embed, self.residual_proj_rel(rel_embed))

        if self.bias_ent is not None: # True
            update_ent_embed += self.bias_ent

        if self.bias_rel is not None: # True
            update_rel_embed += self.bias_rel

        # update_ent_embed = self.bn_ent(update_ent_embed)
        # update_rel_embed = self.bn_rel(update_rel_embed)

        if self.activation is None:
            return update_ent_embed, update_rel_embed
        else:
            return self.activation(update_ent_embed), self.activation(update_rel_embed)




    # def aggregate_entities(self, sub_ent_embed_proj, obj_ent_embed_proj, edge_index, edge_type, ent_embed, num_rel):
    #     b = getattr(self, 'b')
    #
    #     sub_ent_index = edge_index[self.sub_ent_dim]
    #     obj_ent_index = edge_index[self.obj_ent_dim]
    #
    #     # shape = (E, FIN)
    #     sub_embeding = sub_ent_embed_proj.index_select(self.ent_dim, sub_ent_index)
    #     obj_embeding = obj_ent_embed_proj.index_select(self.ent_dim, obj_ent_index)
    #     so_embeding = sub_embeding + obj_embeding
    #     so_embed = self.leakyReLU(so_embeding)
    #
    #     # shape = (E)
    #     scores_per_edge = (so_embed * b).sum(dim=-1)
    #     # shape = (E, 1)
    #     attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_type, num_rel)
    #     attentions_per_edge = self.dropout(attentions_per_edge)
    #
    #     # shape = (E, FIN) * (E) -> (E, FIN)
    #     attentions_embed = so_embeding * attentions_per_edge
    #
    #     # shape = (R, FIN)
    #     out_rel_embed = self.neighborhood_aggregation(attentions_embed, ent_embed, edge_type, num_rel)
    #
    #     return self.kernel_rel(out_rel_embed)

    def aggregate_neighbors(self, sub_ent_embed_proj, obj_ent_embed_proj, rel_embed_proj, edge_index, edge_type, ent_embed, num_ent, num_rel, mode):

        # Step 2: Edge attention calculation

        kernel = getattr(self, 'kernel_{}'.format(mode))

        a = getattr(self, 'a')
        # b = getattr(self, 'b')
        if mode == 'in':
            edge_type_index = edge_type
        else:
            edge_type_index = edge_type-num_rel

        sub_ent_index = edge_index[self.sub_ent_dim]
        obj_ent_index = edge_index[self.obj_ent_dim]

        # shape = (E, FIN)
        sub_embeding = sub_ent_embed_proj.index_select(self.ent_dim, sub_ent_index)
        obj_embeding = obj_ent_embed_proj.index_select(self.ent_dim, obj_ent_index)
        edge_embedding = rel_embed_proj.index_select(self.rel_dim, edge_type)
        triple_embedding = sub_embeding + obj_embeding + edge_embedding
        # triple_embed = self.leakyReLU(triple_embedding)

        # so_embeding = sub_embeding + obj_embeding
        # so_embed = self.leakyReLU(so_embeding)


        # shape = (E)
        scores_per_edge = self.leakyReLU((triple_embedding * a).sum(dim=-1))
        # scores_per_edge_r = (triple_embed * b).sum(dim=-1)

        # shape = (E, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, sub_ent_index, num_ent)
        attentions_per_edge = self.dropout(attentions_per_edge)

        attentions_per_edge_r = self.neighborhood_aware_softmax(scores_per_edge, edge_type_index, num_rel)
        attentions_per_edge_r = self.dropout(attentions_per_edge_r)



        # Step 3: Neighborhood aggregation
        # shape = (E, FIN) * (E) -> (E, FIN)
        attentions_embed = triple_embedding * attentions_per_edge
        attentions_embed_r = triple_embedding * attentions_per_edge_r

        # shape = (N, FIN)
        out_ent_embed = self.neighborhood_aggregation(attentions_embed, ent_embed, sub_ent_index, num_ent)

        # shape = (R, FIN)
        out_rel_embed = self.neighborhood_aggregation(attentions_embed_r, ent_embed, edge_type_index, num_rel)

        # shape = (E, FIN) * (FIN, FOUT) = (E, FOUT) # shape = (R, FIN) * (FIN, FOUT) = (R, FOUT)
        return kernel(out_ent_embed), self.kernel_rel(out_rel_embed)



    def neighborhood_aware_softmax(self, scores_per_edge, sub_ent_index, num_ent):

        # Calculate the numerator
        # (E)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, sub_ent_index, num_ent)

        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E) -> (E, 1)
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, sub_ent_index, num_ent):
        # shape = (E)
        sub_ent_index_broadcasted = self.explicit_broadcast(sub_ent_index, exp_scores_per_edge)

        # shape = (N)
        size = list(exp_scores_per_edge.shape)
        size[self.ent_dim] = num_ent
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        neighborhood_sums.scatter_add_(self.ent_dim, sub_ent_index_broadcasted, exp_scores_per_edge)

        # shape = (N) -> (E)
        return neighborhood_sums.index_select(self.ent_dim, sub_ent_index)

    def neighborhood_aggregation(self, attentions_embed, ent_embed, sub_ent_index, num_ent):
        size = list(attentions_embed.shape)
        size[self.ent_dim] = num_ent # num_ent
        out_ent_embed = torch.zeros(size, dtype=ent_embed.dtype, device=ent_embed.device)

        # shape = (E) -> (E, FOUT)
        sub_ent_index_broadcasted = self.explicit_broadcast(sub_ent_index, attentions_embed)

        # shape = (E, FOUT) -> (N, FOUT)
        out_ent_embed.scatter_add_(self.ent_dim, sub_ent_index_broadcasted, attentions_embed)

        return out_ent_embed

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)










