# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fairseq import metrics, utils

import numpy as np
from torch import nn
import torch.nn.functional as F
import h5py

from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('d2gpo_label_smoothed_cross_entropy')
class D2GPoLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, label_smoothing,
                 d2gpo_weight_path, d2gpo_vocab_path, d2gpo_alpha, d2gpo_temperature, d2gpo_criterion, d2gpo_post_softmax,
                 ignore_prefix_size=0, report_accuracy=False):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        # load the pretrained weight
        assert d2gpo_weight_path is not None
        assert d2gpo_vocab_path is not None
        
        fw = h5py.File(d2gpo_weight_path, 'r')
        lw = fw['weights']
        self.d2gpo_weights = np.array(lw)
        fw.close()

        with open(d2gpo_vocab_path, 'r', encoding='utf-8') as fin:
            data = fin.readlines()
        self.d2gpo_vocab = [line.strip() for line in data if len(line.strip())>0]

        assert len(task.target_dictionary) == self.d2gpo_weights.shape[0] \
               and self.d2gpo_weights.shape[0] == self.d2gpo_weights.shape[1] \
               and self.d2gpo_weights.shape[0] == len(self.d2gpo_vocab)

        # check the vocabulary
        for widx in range(len(task.target_dictionary)):
            assert task.target_dictionary.symbols[widx] == self.d2gpo_vocab[widx]

        self.d2gpo_alpha = d2gpo_alpha
        self.d2gpo_temperature = d2gpo_temperature

        self.d2gpo_criterion_ = d2gpo_criterion
        if self.d2gpo_criterion_ == 'wassdistance':
            self.d2gpo_criterion = SinkhornDistance(eps=0.00001, max_iter=10, reduction='none')
        else:
            self.d2gpo_criterion = nn.KLDivLoss(reduction='none')

        self.d2gpo_post_softmax = d2gpo_post_softmax

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # params for D2GPo loss
        parser.add_argument('--d2gpo-alpha', default=0.1, type=float,
                            help='d2gpo alpha')
        parser.add_argument('--d2gpo-temperature', default=2.0, type=float,
                            help='d2gpo temperature')
        parser.add_argument('--d2gpo-weight-path', type=str,
                            help='d2gpo weight path')
        parser.add_argument('--d2gpo-vocab-path', type=str,
                            help='d2gpo vocabulary path')
        parser.add_argument('--d2gpo-post-softmax', action="store_true",
                            help='d2gpo post softmax')
        parser.add_argument('--d2gpo-criterion', type=str,
                            help='d2gpo criterion')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss, kd_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'kd_loss': kd_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output['n_correct'] = utils.item(n_correct.data)
            logging_output['total'] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        
        T = self.d2gpo_temperature
        
        target_weights = torch.from_numpy(self.d2gpo_weights[target.squeeze(-1).cpu()]).type_as(lprobs)
        
        if self.d2gpo_post_softmax:
            if self.d2gpo_criterion_ == 'wassdistance':
                kd_loss = self.d2gpo_criterion(input=lprobs.exp(), 
                                    target=F.softmax(target_weights / T, dim=-1))
            else:
                kd_loss = self.d2gpo_criterion(input=lprobs, 
                                    target=F.softmax(target_weights / T, dim=-1))
        else:
            if self.d2gpo_criterion_ == 'wassdistance':
                kd_loss = self.d2gpo_criterion(input=lprobs.exp(), 
                                    target=target_weights)
            else:
                kd_loss = self.d2gpo_criterion(input=lprobs, 
                                target=target_weights)
        kd_loss = kd_loss.sum(dim=-1, keepdim=True)

        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        
        pad_mask = target.eq(self.padding_idx)
        kd_loss.masked_fill_(pad_mask, 0.)
        kd_loss = kd_loss.sum()

        loss = loss * (1. - self.d2gpo_alpha) + kd_loss * self.d2gpo_alpha * T * T

        return loss, nll_loss, kd_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask)))
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        kd_loss_sum = sum(log.get('kd_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('kd_loss', kd_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

        total = utils.item(sum(log.get('total', 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar('total', total)
            n_correct = utils.item(
                sum(log.get('n_correct', 0) for log in logging_outputs)
            )
            metrics.log_scalar('n_correct', n_correct)
            metrics.log_derived(
                'accuracy',
                lambda meters: round(
                    meters['n_correct'].sum * 100.0 / meters['total'].sum, 3
                ) if meters['total'].sum > 0 else float('nan'),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, input, target):

        x = input
        y = target

        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost#, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        print(x.size(), y.size())
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        print(x_col.size(), y_lin.size())
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        print(C.size())
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1