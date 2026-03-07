import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging


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

            # print(f'Initialized e_p_{e} with {p}')
            # print(f'Initialized e_k_{e} with {k}')
            # print(f'Initialized e_a_{e} with {a}')

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
                logging.warning(f"NaN detected in K, A, or p for layer {l}")
            # print("A has nan:", torch.isnan(A).any())
            # print("K has nan:", torch.isnan(K).any())
            # print("p has nan:", torch.isnan(p).any())

            # print("A nan rows:", torch.isnan(A).any(dim=1).nonzero())
            # print("K nan rows:", torch.isnan(K).any(dim=1).nonzero())

            # logging.info("x_querry: {}".format(x_querry.cpu().detach().numpy()))

            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)

            # logging.info("a_querry: {}".format(a_querry.cpu().detach().numpy()))

            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            aq_k = F.relu(-aq_k)
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # logging.info("P_: {}".format(P_.cpu().detach().numpy()))
        else:
            P_ = None

        return P_
