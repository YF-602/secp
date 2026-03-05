import timm
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from utils.inc_net import BaseNet, get_backbone
import math
from torch.nn import functional as F
from .zoo import *
import logging

def build_promptmodel(modelname='vit_base_patch16_224', Prompt_Token_num=10, VPT_type="Deep", args=None):
    basic_model = timm.create_model(modelname, pretrained=True)#pretrained_cfg_overlay=dict(file=args['model_path'])
    if modelname in ['vit_base_patch16_224']:
        model = VPT_ViT(Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type, args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(modelname))

    # drop head.weight and head.bias
    basicmodeldict = basic_model.state_dict()
    basicmodeldict.pop('head.weight')
    basicmodeldict.pop('head.bias')

    model.load_state_dict(basicmodeldict, False)

    model.head = torch.nn.Identity()

    model.Freeze()

    return model


class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=200, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, Prompt_Token_num=1,
                 VPT_type="Shallow", basic_state_dict=None, args=None):

        # Recreate ViT
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer)

        print('Using VPT model')
        self.args = args
        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.VPT_type = VPT_type
        if VPT_type == "Deep":
            print("Using Deep Prompt")
            # self.TSP = DPrompt(args, 768, args["nb_tasks"] - 1, Prompt_Token_num / 2)
            self.RSP = NDPrompt(args, 768, 1, round(args["init_cls"] * self.args["prompt_pool_num"]), Prompt_Token_num / 2)
            # self.TIP = torch.zeros(depth, int(Prompt_Token_num/2), embed_dim) # 记录RSP，用于导出TSP
            self.register_buffer("TIP",torch.zeros(depth, int(Prompt_Token_num/2), embed_dim))
            self.Avg_TSP = torch.zeros(depth, int(Prompt_Token_num/2), embed_dim)  
            self.Prompt_Encoder = PROMPT_Encoder(args, depth, prompt_length=int(Prompt_Token_num/2), prompt_featuers=embed_dim)

        else:  # "Shallow"
            print("Using Shallow Prompt")
            # self.TSP = DPrompt(768, num_classes / 2, Prompt_Token_num / 2)
            self.RSP = NDPrompt(768, 1, 20, Prompt_Token_num / 2)
            # self.TIP = torch.zeros(1, int(Prompt_Token_num/2), embed_dim)
            self.register_buffer("TIP",torch.zeros(1, int(Prompt_Token_num/2), embed_dim))
            self.Avg_TSP = torch.zeros(1, int(Prompt_Token_num/2), embed_dim) 
            self.Prompt_Encoder = PROMPT_Encoder(args, 1, prompt_length=int(Prompt_Token_num/2), prompt_featuers=embed_dim)

        self.Prompt_Token_num = Prompt_Token_num

    def New_CLS_head(self, new_classes=15):
        self.head = nn.Linear(self.embed_dim, new_classes)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        # self.TIP.requires_grad = True
        try:
            for param in self.Prompt_Encoder.fc_mu.parameters():
                param.requires_grad = True
            for param in self.Prompt_Encoder.fc_std.parameters():
                param.requires_grad = True
            for param in self.RSP.parameters():
                param.requires_grad = True
        except:
            pass

    def Freeze_new(self):
        for param in self.parameters():
            param.requires_grad = False

        # self.TIP.requires_grad = True
        try:
            for param in self.Prompt_Encoder.fc_mu.parameters():
                param.requires_grad = True
            for param in self.Prompt_Encoder.fc_std.parameters():
                param.requires_grad = True
        except:
            print("Prompt_Encoder cant grad")
            pass

    def obtain_prompt(self):
        return 0

    def load_prompt(self, prompt_state_dict):
        pass

    def forward_features(self, x,targets=None,train=False,proto=False,perturb_var=0,cur_class=None):
        # tsp, (mu, std) = self.Prompt_Encoder(x, self.TIP, perturb_var)  

        # x_fea = self.Prompt_Encoder.prompt_backbone(x)
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        num_tokens = x.shape[1]
        KL = 0

        loss_match=0
        if self.VPT_type == "Deep":
            for i in range(len(self.blocks)):
                x_query = x[:, 0, :]

                # logging.info("x_query: {}".format(x_query[:5]))

                RSP = self.RSP.forward(x_query, i, train=train, proto=proto)

                # logging.info("RSP: {}".format(RSP[:5]))

                self.TIP[i] = RSP.mean(0).detach()
                tsp, kl = self.Prompt_Encoder(x, self.TIP, i, perturb_var) 
                KL += kl
                TSP = self.args["avg_alpha"] * self.Avg_TSP[i].expand(x.shape[0], -1, -1).to(x.device) + (1-self.args["avg_alpha"]) * tsp
                # TSP,loss = self.TSP.forward(x_query, i, targets=targets)
                if targets is not None:
                    # 删除anchor样本对应的TSP
                    TSP_tmp = TSP
                    targets_tmp = targets
                    if cur_class is not None:
                        TSP_tmp = TSP_tmp[:-len(cur_class)]
                        targets_tmp = targets_tmp[:-len(cur_class)]
                    if targets.shape[0] != TSP_tmp.shape[0]:
                        targets_tmp = targets_tmp.repeat(int(TSP_tmp.shape[0] / targets_tmp.shape[0]))
                    loss = prompt_centloss(TSP_tmp, targets_tmp)
                    loss_match+=loss

                # 先拼接TSP，再拼接RSP，因为RSP的维度可能和TSP不同，先拼接TSP可以保证RSP的维度正确
                if TSP is not None:
                    # print(TSP.shape)
                    x = torch.cat([x, TSP], dim=1)
                if RSP is not None:
                    # print(TSP.shape)
                    if x.shape[0]!=RSP.shape[0]:
                        if train or proto:
                            x=x.repeat(int(RSP.shape[0]/x.shape[0]),1,1)
                    x = torch.cat([x, RSP], dim=1)
                x = self.blocks[i](x)[:, :num_tokens]

                # logging.info("{} blocks output x {}".format(i, x))
                # logging.info("block {}".format(i))
                # logging.info("x nan: {}".format(torch.isnan(x).any()))
                # logging.info("TSP nan: {}".format(torch.isnan(TSP).any()))
                # logging.info("RSP nan: {}".format(torch.isnan(RSP).any()))
                # logging.info("x max: {}".format(x.abs().max()))
                # logging.info("TSP max: {}".format(TSP.abs().max()))
                # logging.info("RSP max: {}".format(RSP.abs().max()))
            
        else:  # self.VPT_type == "Shallow"
            print("Can only use Deep vpt type!")

        x = self.norm(x)
        return x,loss_match/len(self.RSP.e_layers),KL

    def forward(self, x, targets=None,train=False,proto=False, perturb_var=0,cur_class=None):
        x,loss_match,kl = self.forward_features(x, targets=targets,train=train, proto=proto, perturb_var=perturb_var, cur_class=cur_class)
        x = x[:, 0, :]
        return x,loss_match,kl

class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            fc.old_out=nb_output
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc


    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        x,loss_match,kl=self.backbone(x)
        return x

    def forward(self, x, targets=None,train=False, proto=False, perturb_var=0, cur_class=None):
        x,loss_match,kl = self.backbone(x, targets=targets,train=train, proto=proto, perturb_var=perturb_var, cur_class=cur_class)

        # logging.info("features: {}".format(x[:5]))

        out = self.fc(x)

        # logging.info("logits: {}".format(out["logits"]))

        out.update({"features": x})
        out.update({"loss_match": loss_match})
        out.update({"kl": kl })
        # out["kl"] = out.get("kl", 0) + kl
        return out

class PROMPT_Encoder(nn.Module):
    def __init__(self, args, depth, prompt_length, prompt_featuers=768):
        super(PROMPT_Encoder, self).__init__()
        self.prompt_length = prompt_length
        self.prompt_featuers = prompt_featuers
        self.depth = depth
        self.prompt_featuers=prompt_featuers

        newargs=copy.deepcopy(args)
        newargs['backbone_type']=newargs['backbone_type'].replace('_vpt','')
        self.prompt_backbone = get_backbone(newargs)

        self.fc_mu = nn.Sequential(
            nn.Linear(prompt_featuers*2, 256),
            nn.Linear(256, prompt_length*prompt_featuers)
        )
        self.fc_std = nn.Sequential(
            nn.Linear(prompt_featuers*2, 256),
            nn.Linear(256, prompt_length*prompt_featuers)
        )

    def forward(self, x, tip, i=0, perturb_var=0):
        # 新增i为层数
        # x是blocks或prompt_backbone的输出
        bs = x.size(0)
        if x.dim() == 2: # prompt_backbone输出
            fea_x = x
        elif x.dim() == 3: # blocks输出
            fea_x = x[:, 0, :].reshape(-1, self.prompt_featuers)  # 直接取CLS token
        else:
            raise ValueError("Invalid input dimension for PROMPT_Encoder, expected 2 or 3, got {}".format(x.dim()))
        # tip = tip.detach()[i,0,:].expand(bs, -1, -1).reshape(-1, self.prompt_featuers)   
        tip = tip.detach()[i, 0, :].unsqueeze(0).expand(bs, -1) 
        # fea_x = fea_x.reshape(-1, self.prompt_featuers)         
        fea = torch.cat([tip, fea_x], dim=1)

        mu = self.fc_mu(fea)
        std = F.softplus(self.fc_std(fea)-5, beta=1)
        prompt = self.reparameterise(mu, std, perturb_var)

        # prompt = prompt.reshape(bs, self.depth, self.prompt_length, self.prompt_featuers)
        prompt = prompt.reshape(bs, self.prompt_length, self.prompt_featuers)
        # 直接计算kl
        kl = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1) / mu.size(0)

        return prompt, kl

        
    def reparameterise(self, mu, std, perturb_var):
        eps = torch.randn_like(std)*perturb_var
        return mu + std*eps

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.old_out=0
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if self.old_out==0:
            weight=self.weight
            # out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            weight=torch.cat((self.weight[:self.old_out].detach().clone(),self.weight[self.old_out:,]))
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(weight, p=2, dim=1))
        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}

def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)