import numpy as np
import torch
import torch.nn.functional as F
# from torch_geometric.utils import to_dense_batch
from utils.utils_doctor import my_to_dense_batch


class HookModule:
    def __init__(self, model, module):
        self.model = model
        self.activations = None

        module.register_forward_hook(self._hook_activations)

    def _hook_activations(self, module, inputs, outputs):
        self.activations = outputs

    def grads(self, outputs, inputs, retain_graph=True, create_graph=True):
        grads = torch.autograd.grad(outputs=outputs,
                                    inputs=inputs,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph,
                                    allow_unused=True)[0]
        self.model.zero_grad()
        return grads


class GradConstraint_CLS:

    def __init__(self, model, modules, channel_paths, device):
        print('- Grad Constraint')
        self.modules = []
        self.channels = []

        for module in modules:
            self.modules.append(HookModule(model=model, module=module))
        for channel_path in channel_paths:
            self.channels.append(torch.from_numpy(np.load(channel_path)).to(device))

    def loss_channel(self, outputs, labels, batch):
        # high response channel loss
        # probs = torch.argsort(-outputs, dim=1)

        # labels_ = torch.zeros_like(labels, dtype=labels.dtype)
        # labels_[probs[:,0] == labels] = probs[probs[:,0] == labels, 1]
        # labels_[probs[:,0] != labels] = probs[probs[:,0] != labels, 0]
        # nll_loss_ = torch.nn.NLLLoss()(outputs, labels_)

        # low response channel loss
        # channel_loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        # channel_loss = torch.nn.NLLLoss()(outputs, labels)
        # probs = torch.argsort(-outputs, dim=1)
        # outs = outputs[:,labels]
        outs_ = torch.zeros_like(labels, dtype=outputs.dtype)
        outs_[labels==0] = outputs[labels==0, 0]
        outs_[labels==1] = outputs[labels==1, 1]
        # channel_loss = 

        channel_loss = torch.nn.NLLLoss()(outputs, labels)
        # channel_loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        loss = 0
        for i, module in enumerate(self.modules):
            # high response channel loss
            loss += _loss_channel(heat_mask=self.channels[0],
                                  heat_matrix=self.channels[1],
                                  grads=module.grads(outputs=-channel_loss,
                                                     inputs=module.activations),
                                  labels=labels,
                                  batch = batch,
                                  is_high=True)

            # low response channel loss
            loss += _loss_channel(heat_mask=self.channels[0],
                                  heat_matrix=self.channels[1],
                                  grads=module.grads(outputs=-channel_loss,
                                                     inputs=module.activations),
                                  labels=labels,
                                  batch = batch,
                                  is_high=False)
        return loss


def _loss_channel(heat_mask, heat_matrix, grads, labels, batch, is_high=True):
    grads = torch.abs(grads)
    # grads = F.relu(grads)
    # channel_grads = torch.sum(grads, dim=(2, 3))  # [batch_size, channels]
    batch_list = batch
    batch_size = len(batch_list)
    # batch_list = torch.Tensor(batch_list).long().to(grads.device)
    batch_index = torch.arange(batch_size).to(grads.device).repeat_interleave(batch_list)
    # batch_index = batch_index.view((-1,) + (1,) * (grads.dim() - 1)).expand_as(grads)

    channel_grads, mask, num_nodes = my_to_dense_batch(batch, grads, batch_index)
    channel_grads = torch.sum(channel_grads, dim=1)
    channel_grads = channel_grads/(num_nodes.view(channel_grads.shape[0],1).repeat([1,channel_grads.shape[1]]))
    # loss = 0
    # if is_high:
    #     for b, l in enumerate(labels):
    #         loss += (channel_grads[b] * channels[l]).sum()
    # else:
    #     for b, l in enumerate(labels):
    #         loss += (channel_grads[b] * (1 - channels[l])).sum()
    # loss = loss / labels.size(0)


    channel_mask = torch.zeros_like(channel_grads, dtype = heat_mask.dtype)
    channel_matrix = torch.zeros_like(channel_grads, dtype = heat_matrix.dtype)
    for i in range(labels.max() + 1):
        index_bool = (labels==i)
        channel_mask[index_bool,:] = heat_mask[i].repeat(index_bool.sum(),1)
        channel_matrix[index_bool,:] = heat_matrix[i].repeat(index_bool.sum(),1)
    if is_high:
        channel_mask = channel_mask
        loss = torch.sum((channel_matrix - channel_grads)*channel_mask, dim=(0,1))
    else:
        channel_mask = 1 - channel_mask
        loss = torch.sum(channel_grads * channel_mask, dim=(0,1))
    loss = loss / labels.size(0)

    return loss

