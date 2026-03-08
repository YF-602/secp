import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
from SupConLoss import SupConLoss, CenterLoss, prompt_con_loss, prompt_centloss


def tensor_prompt(a, b, c=None, ortho=False, grad=True):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, int(b)), requires_grad=grad)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=grad)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


class DPrompt(nn.Module):
    def __init__(self, args, emb_d, nb_task, e_p_length, key_dim=768):
        super().__init__()
        self.task_count = -1
        self.emb_d = emb_d
        self.key_d = key_dim
        self.args=args
        self.last_class_num = 0
        self.class_num = 0
        self.prompt_num = 0
        self.last_prompt_num = 0
        self.ini_class = args["init_cls"]
        self.incre_class = args["increment"]
        self._init_smart(nb_task, e_p_length)
        self.supconloss = SupConLoss()

        # e prompt init
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    def _init_smart(self, nb_task, e_p_length, ortho_mu=0):

        # prompt basic param
        self.nb_task = int(nb_task)
        self.e_p_length = int(e_p_length)

        # 数量等于blocks数量12，详见VPT_ViT类的定义depth=12
        self.e_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] 

        self.e_pool_size = round(self.ini_class * self.args["prompt_pool_num"]) + nb_task * round(self.incre_class * self.args["prompt_pool_num"])
        # strenth of ortho penalty
        self.ortho_mu = ortho_mu

    def updatek(self, proto, class_index):

        for l in self.e_layers:
            K = getattr(self, f'e_k_{l}')
            K[class_index] = proto[l]
            setattr(self, f'e_k_{l}', K)

    def process_task_count(self, class_num):

        self.last_class_num = self.class_num
        self.class_num = class_num
        self.task_count += 1
        self.last_prompt_num = self.prompt_num
        if self.task_count == 0:
            self.prompt_num = round(self.ini_class * self.args["prompt_pool_num"])
        else:
            self.prompt_num = self.prompt_num + round(self.incre_class * self.args["prompt_pool_num"])
        print(self.last_class_num)
        print(self.class_num)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point

        for k in range(0, self.e_pool_size):
            # print(k)
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            # print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(0, self.e_pool_size):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(self, x_querry, l):

        if l in self.e_layers:

            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')
            if torch.isnan(K).any() or torch.isnan(A).any() or torch.isnan(p).any():
                logging.warning(f"DPrompt NaN detected in K, A, or p for layer {l}")

            if self.task_count > 0:
                K = torch.cat((K[:self.last_prompt_num].detach().clone(), K[self.last_prompt_num:self.prompt_num]),
                              dim=0)
                A = torch.cat((A[:self.last_prompt_num].detach().clone(), A[self.last_prompt_num:self.prompt_num]),
                              dim=0)
                p = torch.cat((p[:self.last_prompt_num].detach().clone(), p[self.last_prompt_num:self.prompt_num]),
                              dim=0)
            else:
                K = K[0:self.prompt_num]
                A = A[0:self.prompt_num]
                p = p[0:self.prompt_num]

            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            aq_k = torch.nn.functional.softmax(aq_k)
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

        else:
            P_ = None

        return P_

    def dismatch_loss(self, x_querry):

        loss = 0.0
        for i in self.e_layers:
            K = getattr(self, f'e_k_{i}')
            A = getattr(self, f'e_a_{i}')
            p = getattr(self, f'e_p_{i}')
            K = torch.cat((K[:self.last_prompt_num].detach().clone(), K[self.last_prompt_num:self.prompt_num]), dim=0)
            A = torch.cat((A[:self.last_prompt_num].detach().clone(), A[self.last_prompt_num:self.prompt_num]), dim=0)
            p = torch.cat((p[:self.last_prompt_num].detach().clone(), p[self.last_prompt_num:self.prompt_num]), dim=0)
            x_query = x_querry[i - self.e_layers[0], :, :].squeeze()
            # print(x_query.shape)
            a_query = torch.einsum('bd,kd->bkd', x_query, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_query, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # print(aq_k)
            k_idx = range(self.last_prompt_num, self.prompt_num)
            loss += abs(aq_k[:, k_idx]).sum() / x_query.shape[0]
        loss = loss / len(self.e_layers)
        # print(loss)
        return loss


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean()


# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class NDPrompt(nn.Module):
    def __init__(self, ags, emb_d, n_tasks, e_pool_size, e_p_length, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.args=ags
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(e_pool_size, e_p_length)
        # e prompt init
        for e in self.e_layers:
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)

            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    def _init_smart(self, e_pool_size, e_p_length):

        # prompt basic param
        self.e_pool_size = int(e_pool_size)
        self.e_p_length = int(e_p_length)
        
        # 数量等于blocks数量12，详见VPT_ViT类的定义depth=12
        self.e_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] 

    def process_task_count(self):
        self.task_count += 1

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            # 在u方向上投影v
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()

            # if uk.norm() < 1e-8:
            #     print('zero norm!!!')

            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(self, x_querry, l):

        if l in self.e_layers:
            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')

            if torch.isnan(K).any() or torch.isnan(A).any() or torch.isnan(p).any():
                logging.warning(f"NDPrompt NaN detected in K, A, or p for layer {l}")

            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)

            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            aq_k = F.relu(-aq_k)
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

        else:
            P_ = None

        return P_
