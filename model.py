# coding: UTF-8
import torch
import torch.nn as nn
from transformers import RobertaModel
import torch.nn.functional as F 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size = 768

class Base(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.roberta_model = RobertaModel.from_pretrained(args.model_name).to(device)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
        self.mlp = nn.Linear(2 * args.embedding_size, args.mlp_size).to(device)
        # self.mlp = nn.Sequential(
        #     nn.Linear(2 * args.embedding_size, args.mlp_size).to(device),
        #     nn.Softmax().to(device)
        # )
        self.embed_size = args.embedding_size

    def forward(self, batch_arg, arg_mask, batch_e, batch_size):
        sent_emb = self.roberta_model(batch_arg, arg_mask)[0].to(device)
        event_pair_embed = torch.tensor([]).to(device)
    
        for i in range(batch_size):
            e_emb = self.extract_event(sent_emb[i], batch_e[i])
            if i == 0:
                event_pair_embed = e_emb
            else:
                event_pair_embed = torch.cat((event_pair_embed, e_emb))
        prediction = self.mlp(event_pair_embed)
        return prediction

    def extract_event(self, embed, event_idx):
        e1_start = event_idx[0]
        e1_end = event_idx[1]
        e2_start = event_idx[2]
        e2_end = event_idx[3]
        e1_embed = torch.zeros(1, self.embed_size).to(device)
        e2_embed = torch.zeros(1, self.embed_size).to(device)
        e1_num = e1_end - e1_start
        e2_num = e2_end - e2_start
        for i in range(e1_start, e1_end):
            e1_embed += embed[i]
        for j in range(e2_start, e2_end):
            e2_embed += embed[j]
        e1_embed = e1_embed / e1_num
        e2_embed = e2_embed / e2_num
        event_embed = torch.cat((e1_embed, e2_embed), dim=1).to(device)
        return event_embed

class modified_with_attention_mask(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embed_size = args.embedding_size
        self.num_heads = 8
        self.out_features = 1
        self.alpha = 0.2
        self.sent_len = 196

        self.roberta_model = RobertaModel.from_pretrained(args.model_name).to(device)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
        self.query_generation_layer = nn.Linear(self.embed_size, self.embed_size).to(device)
        self.key_generation_layer = nn.Linear(self.embed_size, self.embed_size).to(device)
        self.value_generation_layer = nn.Linear(self.embed_size, self.embed_size).to(device)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_size, num_heads=self.num_heads).to(device)
        self.W = nn.Linear(self.embed_size, self.out_features)
        self.a_T = nn.Linear(2 * self.out_features, 1, bias = False)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_T.weight)
        self.event_mlp = nn.Linear(2 * args.embedding_size, args.mlp_size).to(device)
        self.sentence_mlp = nn.Linear(self.sent_len, args.mlp_size).to(device)
        # self.sentence_mlp = nn.Sequential(
        #     nn.Linear(self.sent_len, 64).to(device),
        #     nn.ReLU().to(device),
        #     nn.Linear(64,32).to(device),
        #     nn.ReLU().to(device),
        #     nn.Linear(32,args.mlp_size).to(device)        
        #     )

    def forward(self, batch_arg, arg_mask, batch_e, batch_size):
        sent_emb = self.roberta_model(batch_arg, arg_mask)[0].to(device)

        prediction = 0.85 * self.event_level_prediction(sent_emb,batch_e,batch_size) \
            + 0.15 * self.sentence_level_prediction(sent_emb)
        return prediction
    
    def sentence_level_prediction(self, sent_emb):
        Q, K, V = self.generate_key(sent_emb)
        attn_output, attn_output_weights = self.multihead_attn(Q, K, V)
        for i in range(sent_emb.size(0)):
            event_mask = attn_output_weights[i]
            N = sent_emb[i].size(0)
            Wh = self.W(sent_emb[i])
            H1 = Wh.unsqueeze(1).repeat(1,N,1)
            H2 = Wh.unsqueeze(0).repeat(N,1,1)
            attn_input = torch.cat([H1, H2], dim = -1)

            e = F.leaky_relu(self.a_T(attn_input).squeeze(-1), negative_slope = self.alpha) # [N, N]
            
            attn_mask = -1e18*torch.ones_like(e)
            masked_e = torch.where(event_mask > 0, e, attn_mask)
            attn_scores = F.softmax(masked_e, dim = -1) # [N, N]

            h_prime = torch.mm(attn_scores, Wh) # [N, F']
            h_prime = F.elu(h_prime) # [N, F']        return prediction
            if i == 0:
                embed = h_prime.permute(1,0)
            else:
                embed = torch.cat((embed, h_prime.permute(1,0)))

        prediction = self.sentence_mlp(embed)
        return prediction


    def generate_key(self, sent_emb):
        Q = self.query_generation_layer(sent_emb).permute(1,0,2)
        K = self.key_generation_layer(sent_emb).permute(1,0,2)
        V = self.value_generation_layer(sent_emb).permute(1,0,2)
        return Q, K, V

    def event_level_prediction(self, sent_emb, batch_e, batch_size):
        event_pair_embed = torch.tensor([]).to(device)
    
        for i in range(batch_size):
            e_emb = self.extract_event(sent_emb[i], batch_e[i])
            if i == 0:
                event_pair_embed = e_emb
            else:
                event_pair_embed = torch.cat((event_pair_embed, e_emb))
        prediction = self.event_mlp(event_pair_embed)
        return prediction

    def extract_event(self, embed, event_idx):
        e1_start = event_idx[0]
        e1_end = event_idx[1]
        e2_start = event_idx[2]
        e2_end = event_idx[3]
        e1_embed = torch.zeros(1, self.embed_size).to(device)
        e2_embed = torch.zeros(1, self.embed_size).to(device)
        e1_num = e1_end - e1_start
        e2_num = e2_end - e2_start
        for i in range(e1_start, e1_end):
            e1_embed += embed[i]
        for j in range(e2_start, e2_end):
            e2_embed += embed[j]
        e1_embed = e1_embed / e1_num
        e2_embed = e2_embed / e2_num
        event_embed = torch.cat((e1_embed, e2_embed), dim=1).to(device)
        return event_embed

class modified(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.roberta_model = RobertaModel.from_pretrained(args.model_name).to(device)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
        self.mlp = nn.Linear(2 * args.embedding_size, args.mlp_size).to(device)
        self.full_knowledge = nn.Linear(196 * args.embedding_size, args.mlp_size).to(device)
        self.embed_size = args.embedding_size

    def forward(self, batch_arg, arg_mask, batch_e, batch_size):
        sent_emb = self.roberta_model(batch_arg, arg_mask)[0].to(device)
        event_pair_embed = torch.tensor([]).to(device)
        sent_emb_cat = self.sent_process(sent_emb)

        for i in range(batch_size):
            e_emb = self.extract_event(sent_emb[i], batch_e[i])
            if i == 0:
                event_pair_embed = e_emb
            else:
                event_pair_embed = torch.cat((event_pair_embed, e_emb))
        prediction = 0.9 * self.mlp(event_pair_embed) + 0.1 * self.full_knowledge(sent_emb_cat)
        return prediction

    def sent_process(self, embed):
        sent_emb_cat = torch.tensor([]).to(device) 
        for i, emb_item in enumerate(embed):
            emb_item_cat = torch.tensor([]).to(device)
            for j,character in enumerate(emb_item):
                if j == 0:
                    emb_item_cat = torch.unsqueeze(character,0)
                else:
                    emb_item_cat = torch.cat((emb_item_cat, torch.unsqueeze(character,0)),dim = 1).to(device)
            if i == 0:
                sent_emb_cat = emb_item_cat
            else:
                sent_emb_cat = torch.cat((sent_emb_cat, emb_item_cat))
        return sent_emb_cat


    def extract_event(self, embed, event_idx):
        e1_start = event_idx[0]
        e1_end = event_idx[1]
        e2_start = event_idx[2]
        e2_end = event_idx[3]
        e1_embed = torch.zeros(1, self.embed_size).to(device)
        e2_embed = torch.zeros(1, self.embed_size).to(device)
        e1_num = e1_end - e1_start
        e2_num = e2_end - e2_start
        for i in range(e1_start, e1_end):
            e1_embed += embed[i]
        for j in range(e2_start, e2_end):
            e2_embed += embed[j]
        e1_embed = e1_embed / e1_num
        e2_embed = e2_embed / e2_num
        event_embed = torch.cat((e1_embed, e2_embed), dim=1).to(device)
        return event_embed

class modified_with_lstm(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.roberta_model = RobertaModel.from_pretrained(args.model_name).to(device)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
        self.hidden_size = 256
        self.embed_size = args.embedding_size
        self.mlp = nn.Linear(2 * args.embedding_size, args.mlp_size).to(device)
        self.full_knowledge_layer = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, batch_first=True)
        self.full_knowledge_pred = nn.Sequential(
            nn.Linear(self.hidden_size, 128).to(device),
            nn.ReLU().to(device),
            nn.Linear(128,64).to(device),
            nn.ReLU().to(device),
            nn.Linear(64,args.mlp_size).to(device)        
            )

    def forward(self, batch_arg, arg_mask, batch_e, batch_size):
        sent_emb = self.roberta_model(batch_arg, arg_mask)[0].to(device)
        event_pair_embed = torch.tensor([]).to(device)
        full_knowledge_information = self.full_knowledge_layer(sent_emb)[0]
        # full_knowledge_embed = torch.squeeze(self.full_knowledge_layer(sent_emb)[1][0],dim=0) 
        full_knowledge_embed = self.full_knowledge_extraction(full_knowledge_information,arg_mask)

        for i in range(batch_size):
            e_emb = self.extract_event(sent_emb[i], batch_e[i])
            if i == 0:
                event_pair_embed = e_emb
            else:
                event_pair_embed = torch.cat((event_pair_embed, e_emb))
        prediction = 0.9 * self.mlp(event_pair_embed) + 0.1 * self.full_knowledge_pred(full_knowledge_embed)
        return prediction

    def full_knowledge_extraction(self, full_knowledge_information, arg_mask):
        for i in range(len(full_knowledge_information)):
            len_sent = torch.count_nonzero(arg_mask[i])
            if i == 0:
                full_knowledge_embed = torch.unsqueeze(full_knowledge_information[i][len_sent - 1],0)
            else:
                full_knowledge_embed = torch.cat((full_knowledge_embed,torch.unsqueeze(full_knowledge_information[i][len_sent - 1],0)),dim=0)
        return full_knowledge_embed
            

    def extract_event(self, embed, event_idx):
        e1_start = event_idx[0]
        e1_end = event_idx[1]
        e2_start = event_idx[2]
        e2_end = event_idx[3]
        e1_embed = torch.zeros(1, self.embed_size).to(device)
        e2_embed = torch.zeros(1, self.embed_size).to(device)
        e1_num = e1_end - e1_start
        e2_num = e2_end - e2_start
        for i in range(e1_start, e1_end):
            e1_embed += embed[i]
        for j in range(e2_start, e2_end):
            e2_embed += embed[j]
        e1_embed = e1_embed / e1_num
        e2_embed = e2_embed / e2_num
        event_embed = torch.cat((e1_embed, e2_embed), dim=1).to(device)
        return event_embed

class GAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        '''
        initialize parameters
        '''
        self.embed_size = args.embedding_size
        self.out_features = 256
        self.alpha = 0.2

        '''
        initialize layers
        '''
        self.roberta_model = RobertaModel.from_pretrained(args.model_name).to(device)
        self.W = nn.Linear(self.embed_size, self.out_features)
        self.a_T = nn.Linear(2 * self.out_features, 1, bias = False)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_T.weight)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
        self.mlp = nn.Linear(self.out_features, args.mlp_size).to(device)

    def forward(self, batch_arg, arg_mask, batch_e, batch_size):
        sent_emb = self.roberta_model(batch_arg, arg_mask)[0].to(device)
        attention_emb = self.attention_cal(sent_emb, batch_e, batch_size)
        prediction = self.mlp(attention_emb)
        return prediction

    def generate_event_mask(self, embed, event_idx):
        node_length = len(embed)
        event_mask = torch.empty(node_length,node_length).fill_(0.1)
        e1_start = event_idx[0]
        e1_end = event_idx[1]
        e2_start = event_idx[2]
        e2_end = event_idx[3]
        for i in range(e1_start, e1_end):
            for j in range(e2_start, e2_end):
                event_mask[i][j] = 0.9
        return event_mask

    def attention_cal(self, sent_emb, batch_e, batch_size):
        event_pair_embed = torch.tensor([]).to(device)
        for i in range(batch_size):
            event_mask = self.generate_event_mask(sent_emb[i], batch_e[i]).to(device)
            N = sent_emb[i].size(0)
            Wh = self.W(sent_emb[i])
            H1 = Wh.unsqueeze(1).repeat(1,N,1)
            H2 = Wh.unsqueeze(0).repeat(N,1,1)
            attn_input = torch.cat([H1, H2], dim = -1)

            e = F.leaky_relu(self.a_T(attn_input).squeeze(-1), negative_slope = self.alpha) # [N, N]
            
            attn_mask = -1e18*torch.ones_like(e)
            masked_e = torch.where(event_mask > 0, e, attn_mask)
            attn_scores = F.softmax(masked_e, dim = -1) # [N, N]

            h_prime = torch.mm(attn_scores, Wh) # [N, F']
            h_prime = F.elu(h_prime) # [N, F']
            if i == 0:
                event_pair_embed = self.avg_pooling(h_prime).unsqueeze(0)
            else:
                event_pair_embed = torch.cat((event_pair_embed, self.avg_pooling(h_prime).unsqueeze(0)))
        return event_pair_embed

    def avg_pooling(self, h_prime):
        for i in range(h_prime.size(0)):
            if i == 0:
                avg = h_prime[i]
            else:
                avg += h_prime[i]
        return avg/h_prime.size(0)

class modified_with_attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_size = args.embedding_size
        self.num_heads = 8

        self.roberta_model = RobertaModel.from_pretrained(args.model_name).to(device)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
        self.query_generation_layer = nn.Linear(self.embed_size, self.embed_size).to(device)
        self.key_generation_layer = nn.Linear(self.embed_size, self.embed_size).to(device)
        self.value_generation_layer = nn.Linear(self.embed_size, self.embed_size).to(device)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_size, num_heads=self.num_heads).to(device)
        self.event_mlp = nn.Linear(2 * args.embedding_size, args.mlp_size).to(device)
        self.sent_mlp = nn.Linear(self.embed_size, args.mlp_size).to(device)

    def forward(self, batch_arg, arg_mask, batch_e, batch_size):
        sent_emb = self.roberta_model(batch_arg, arg_mask)[0].to(device)
        prediction = 0.9 * self.event_level_prediction(sent_emb, batch_e, batch_size) + \
            0.1 * self.sentence_level_prediction(sent_emb) 
        return prediction

    def sentence_level_prediction(self, sent_emb):
        Q, K, V = self.generate_key(sent_emb)
        attn_output, attn_output_weights = self.multihead_attn(Q, K, V)
        # attn_embed = torch.sum(attn_output,dim=0)/attn_output.size(0)
        attn_embed = torch.max(attn_output,dim=0)[0]
        prediction = self.sent_mlp(attn_embed)
        return prediction


    def generate_key(self, sent_emb):
        Q = self.query_generation_layer(sent_emb).permute(1,0,2)
        K = self.key_generation_layer(sent_emb).permute(1,0,2)
        V = self.value_generation_layer(sent_emb).permute(1,0,2)
        return Q, K, V


    def event_level_prediction(self, sent_emb, batch_e, batch_size):
        event_pair_embed = torch.tensor([]).to(device)
        for i in range(batch_size):
            e_emb = self.extract_event(sent_emb[i], batch_e[i])
            if i == 0:
                event_pair_embed = e_emb
            else:
                event_pair_embed = torch.cat((event_pair_embed, e_emb))
        prediction = self.event_mlp(event_pair_embed)
        return prediction
        

    def extract_event(self, embed, event_idx):
        e1_start = event_idx[0]
        e1_end = event_idx[1]
        e2_start = event_idx[2]
        e2_end = event_idx[3]
        e1_embed = torch.zeros(1, self.embed_size).to(device)
        e2_embed = torch.zeros(1, self.embed_size).to(device)
        e1_num = e1_end - e1_start
        e2_num = e2_end - e2_start
        for i in range(e1_start, e1_end):
            e1_embed += embed[i]
        for j in range(e2_start, e2_end):
            e2_embed += embed[j]
        e1_embed = e1_embed / e1_num
        e2_embed = e2_embed / e2_num
        event_embed = torch.cat((e1_embed, e2_embed), dim=1).to(device)
        return event_embed