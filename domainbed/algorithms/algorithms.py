# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import itertools
from typing import List

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import logging
from collections import OrderedDict, defaultdict

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches
from domainbed.models.resnet_mixstyle import resnet18_mixstyle_L234_p0d5_a0d1, resnet50_mixstyle_L234_p0d5_a0d1
from domainbed.models.resnet_mixstyle2 import resnet18_mixstyle2_L234_p0d5_a0d1, resnet50_mixstyle2_L234_p0d5_a0d1
from domainbed.optimizers import get_optimizer
from domainbed.models.kan_adapters import (
    MOEKANLoraInjectedLinear, 
    MOEKANKroneckerInjectedLinear, 
    MOEPureKANInjectedLinear
)
#  import higher

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def to_minibatch(x, y):
    minibatches = list(zip(x, y))
    return minibatches


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        self.featurizer = networks.Featurizer(input_shape, self.hparams, args)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.l_aux = args.l_aux

        self.optimizer = get_optimizer(
            hparams["optimizer"],
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.featurizer.network.require_grad_params = None

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        t = self.predict(all_x)
        loss_aux_list = []
        loss_aux = 0
        if self.l_aux:

            for n, m in self.featurizer.network.named_modules():
                if hasattr(m, 'aux_loss'):
                    loss_aux_list.append(m.aux_loss)

            for layer_loss in loss_aux_list:
                loss_aux += layer_loss

        loss = F.cross_entropy(t, all_y)
        loss += loss_aux
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class Mixstyle(Algorithm):
    """MixStyle w/o domain label (random shuffle)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if hparams["resnet18"]:
            network = resnet18_mixstyle_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class Mixstyle2(Algorithm):
    """MixStyle w/ domain label"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert input_shape[1:3] == (224, 224), "Mixstyle support R18 and R50 only"
        super().__init__(input_shape, num_classes, num_domains, hparams)
        if hparams["resnet18"]:
            network = resnet18_mixstyle2_L234_p0d5_a0d1()
        else:
            network = resnet50_mixstyle2_L234_p0d5_a0d1()
        self.featurizer = networks.ResNet(input_shape, self.hparams, network)

        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = self.new_optimizer(self.network.parameters())

    def pair_batches(self, xs, ys):
        xs = [x.chunk(2) for x in xs]
        ys = [y.chunk(2) for y in ys]
        N = len(xs)
        pairs = []
        for i in range(N):
            j = i + 1 if i < (N - 1) else 0
            xi, yi = xs[i][0], ys[i][0]
            xj, yj = xs[j][1], ys[j][1]

            pairs.append(((xi, yi), (xj, yj)))

        return pairs

    def update(self, x, y, **kwargs):
        pairs = self.pair_batches(x, y)
        loss = 0.0

        for (xi, yi), (xj, yj) in pairs:
            #  Mixstyle2:
            #  For the input x, the first half comes from one domain,
            #  while the second half comes from the other domain.
            x2 = torch.cat([xi, xj])
            y2 = torch.cat([yi, yj])
            loss += F.cross_entropy(self.predict(x2), y2)

        loss /= len(pairs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        return self.network(x)


class ARM(ERM):
    """Adaptive Risk Minimization (ARM)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams["batch_size"]

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class SAM(ERM):
    """Sharpness-Aware Minimization"""

    @staticmethod
    def norm(tensor_list: List[torch.tensor], p=2):
        """Compute p-norm for tensor list"""
        return torch.cat([x.flatten() for x in tensor_list]).norm(p)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        # 1. eps(w) = rho * g(w) / g(w).norm(2)
        #           = (rho / g(w).norm(2)) * g(w)
        grad_w = autograd.grad(loss, self.network.parameters())
        scale = self.hparams["rho"] / self.norm(grad_w)
        eps = [g * scale for g in grad_w]

        # 2. w' = w + eps(w)
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.add_(v)

        # 3. w = w - lr * g(w')
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        # restore original network params
        with torch.no_grad():
            for p, v in zip(self.network.parameters(), eps):
                p.sub_(v)
        self.optimizer.step()

        return {"loss": loss.item()}


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer("update_count", torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams["weight_decay_d"],
            betas=(self.hparams["beta1"], 0.9),
        )

        self.gen_opt = get_optimizer(
            hparams["optimizer"],
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams["weight_decay_g"],
            betas=(self.hparams["beta1"], 0.9),
        )

    def update(self, x, y, **kwargs):
        self.update_count += 1
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])
        minibatches = to_minibatch(x, y)
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat(
            [torch.full((x.shape[0],), i, dtype=torch.int64, device="cuda") for i, (x, y) in enumerate(minibatches)]
        )

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1.0 / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction="none")
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(), [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams["grad_penalty"] * grad_penalty

        d_steps_per_g = self.hparams["d_steps_per_g_step"]
        if self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g:

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {"disc_loss": disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams["lambda"] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {"gen_loss": gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=False,
            class_balance=False,
        )


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(
            input_shape,
            num_classes,
            num_domains,
            hparams,
            conditional=True,
            class_balance=True,
        )


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.0).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        penalty_weight = (
            self.hparams["irm_lambda"] if self.update_count >= self.hparams["irm_penalty_anneal_iters"] else 1.0
        )
        nll = 0.0
        penalty = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams["irm_penalty_anneal_iters"]:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("update_count", torch.tensor([0]))

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 1.0

        nll = 0.0

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx : all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams["vrex_penalty_anneal_iters"]:
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {"loss": loss.item(), "nll": nll.item(), "penalty": penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class OrgMixup(ERM):
    """
    Original Mixup independent with domains
    """

    def update(self, x, y, **kwargs):
        x = torch.cat(x)
        y = torch.cat(y)

        indices = torch.randperm(x.size(0))
        x2 = x[indices]
        y2 = y[indices]

        lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

        x = lam * x + (1 - lam) * x2
        predictions = self.predict(x)

        objective = lam * F.cross_entropy(predictions, y)
        objective += (1 - lam) * F.cross_entropy(predictions, y2)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class CutMix(ERM):
    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def update(self, x, y, **kwargs):
        # cutmix_prob is set to 1.0 for ImageNet and 0.5 for CIFAR100 in the original paper.
        x = torch.cat(x)
        y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(x.size()[0]).cuda()
            target_a = y
            target_b = y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            # compute output
            output = self.predict(x)
            objective = F.cross_entropy(output, target_a) * lam + F.cross_entropy(output, target_b) * (1.0 - lam)
        else:
            output = self.predict(x)
            objective = F.cross_entropy(output, y)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {"loss": objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()

        loss = torch.dot(losses, self.q) / len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams)

    def update(self, x, y, **kwargs):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        minibatches = to_minibatch(x, y)
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = get_optimizer(
                self.hparams["optimizer"],
                #  "SGD",
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # 1. Compute supervised loss for meta-train set
            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # 2. Compute meta loss for meta-val set
            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams["mldg_beta"] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams["mldg_beta"] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)

        self.optimizer.step()

        return {"loss": objective}


#  class SOMLDG(MLDG):
#      """Second-order MLDG"""
#      # This commented "update" method back-propagates through the gradients of
#      # the inner update, as suggested in the original MAML paper.  However, this
#      # is twice as expensive as the uncommented "update" method, which does not
#      # compute second-order derivatives, implementing the First-Order MAML
#      # method (FOMAML) described in the original MAML paper.

#      def update(self, x, y, **kwargs):
#          minibatches = to_minibatch(x, y)
#          objective = 0
#          beta = self.hparams["mldg_beta"]
#          inner_iterations = self.hparams.get("inner_iterations", 1)

#          self.optimizer.zero_grad()

#          with higher.innerloop_ctx(
#              self.network, self.optimizer, copy_initial_weights=False
#          ) as (inner_network, inner_optimizer):
#              for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
#                  for inner_iteration in range(inner_iterations):
#                      li = F.cross_entropy(inner_network(xi), yi)
#                      inner_optimizer.step(li)

#                  objective += F.cross_entropy(self.network(xi), yi)
#                  objective += beta * F.cross_entropy(inner_network(xj), yj)

#              objective /= len(minibatches)
#              objective.backward()

#          self.optimizer.step()

#          return {"loss": objective.item()}


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=(0.001, 0.01, 0.1, 1, 10, 100, 1000)):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= nmb * (nmb - 1) / 2

        self.optimizer.zero_grad()
        (objective + (self.hparams["mmd_gamma"] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {"loss": objective.item(), "penalty": penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning
    (https://arxiv.org/abs/1711.07910)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs * 2, num_classes)
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        self.register_buffer("embeddings", torch.zeros(num_domains, self.featurizer.n_outputs))

        self.ema = self.hparams["mtl_ema"]

    def update(self, x, y, **kwargs):
        minibatches = to_minibatch(x, y)
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding + (1 - self.ema) * self.embeddings[env]

            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = nn.Linear(self.network_f.n_outputs, num_classes)
        # style network
        self.network_s = nn.Linear(self.network_f.n_outputs, num_classes)

        # # This commented block of code implements something closer to the
        # # original paper, but is specific to ResNet and puts in disadvantage
        # # the other algorithms.
        # resnet_c = networks.Featurizer(input_shape, self.hparams)
        # resnet_s = networks.Featurizer(input_shape, self.hparams)
        # # featurizer network
        # self.network_f = torch.nn.Sequential(
        #         resnet_c.network.conv1,
        #         resnet_c.network.bn1,
        #         resnet_c.network.relu,
        #         resnet_c.network.maxpool,
        #         resnet_c.network.layer1,
        #         resnet_c.network.layer2,
        #         resnet_c.network.layer3)
        # # content network
        # self.network_c = torch.nn.Sequential(
        #         resnet_c.network.layer4,
        #         resnet_c.network.avgpool,
        #         networks.Flatten(),
        #         resnet_c.network.fc)
        # # style network
        # self.network_s = torch.nn.Sequential(
        #         resnet_s.network.layer4,
        #         resnet_s.network.avgpool,
        #         networks.Flatten(),
        #         resnet_s.network.fc)

        def opt(p):
            return get_optimizer(hparams["optimizer"], p, lr=hparams["lr"], weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).cuda()

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, x, y, **kwargs):
        all_x = torch.cat([xi for xi in x])
        all_y = torch.cat([yi for yi in y])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {
            "loss_c": loss_c.item(),
            "loss_s": loss_s.item(),
            "loss_adv": loss_adv.item(),
        }

    def predict(self, x):
        return self.network_c(self.network_f(x))


class RSC(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(RSC, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.drop_f = (1 - hparams["rsc_f_drop_factor"]) * 100
        self.drop_b = (1 - hparams["rsc_b_drop_factor"]) * 100
        self.num_classes = num_classes

    def update(self, x, y, **kwargs):
        # inputs
        all_x = torch.cat([xi for xi in x])
        # labels
        all_y = torch.cat([yi for yi in y])
        # one-hot labels
        all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
        # features
        all_f = self.featurizer(all_x)
        # predictions
        all_p = self.classifier(all_f)

        # Equation (1): compute gradients with respect to representation
        all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

        # Equation (2): compute top-gradient-percentile mask
        percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
        percentiles = torch.Tensor(percentiles)
        percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
        mask_f = all_g.lt(percentiles.cuda()).float()

        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f

        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)

        # Section 3.3: Batch Percentage
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu(), self.drop_b)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        mask = torch.logical_or(mask_f, mask_b).float()

        # Equations (3) and (4) again, this time mutting over examples
        all_p_muted_again = self.classifier(all_f * mask)

        # Equation (5): update
        loss = F.cross_entropy(all_p_muted_again, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class KAN_MOA(ERM):
    """KAN-based Mixture of Adapters with enhanced training"""
    
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super().__init__(input_shape, num_classes, num_domains, hparams, args)
        self.args = args
        self.num_domains = num_domains
        
        # Enhanced loss weights from hparams
        self.lipschitz_weight = hparams.get('lipschitz_weight', 0.01)
        self.ortho_weight = hparams.get('ortho_weight', 0.1)
        self.complexity_weight = hparams.get('complexity_weight', 0.1)
        self.load_balance_weight = hparams.get('load_balance_weight', 0.01)
        self.grid_entropy_weight = hparams.get('grid_entropy_weight', 0.01)
        self.domain_routing_weight = hparams.get('domain_routing_weight', 0.1)
        
        # Enable/disable specific losses
        self.enable_load_balance = hparams.get('enable_load_balance', True)
        self.enable_grid_entropy = hparams.get('enable_grid_entropy', True)
        self.enable_pairwise_ortho = hparams.get('enable_pairwise_ortho', True)
        self.enable_domain_routing = hparams.get('enable_domain_routing', False)
        
        # Gradient scaling
        self.kan_grad_scale = hparams.get('kan_init_scale', 0.1)
        
        # Loss tracking for logging
        self.loss_history = defaultdict(list)
        self.training_step = 0
        
        logger.info(f"KAN_MOA initialized with weights - Lip: {self.lipschitz_weight}, "
                   f"Ortho: {self.ortho_weight}, LoadBal: {self.load_balance_weight}, "
                   f"GridEnt: {self.grid_entropy_weight}, DomainRoute: {self.domain_routing_weight}")
    
    def compute_domain_routing_loss(self, adapters, x):
        """
        Domain-aware routing loss: Encourage different domains to use different expert distributions.
        Uses POSITIVE KL divergence with a target, clamped to prevent dominance.
        """
        if not self.enable_domain_routing:
            return torch.tensor(0.0, device=x[0].device)
        
        domain_routing_loss = 0.0
        num_domains = len(x)
        
        if num_domains < 2:
            return torch.tensor(0.0, device=x[0].device)
        
        try:
            batch_size = x[0].shape[0]
            count = 0
            
            for m in adapters:
                if not hasattr(m, 'last_routing_weights') or m.last_routing_weights is None:
                    continue
                
                weights = m.last_routing_weights  # (N_total, num_experts)
                num_experts = weights.shape[-1]
                
                try:
                    # Reshape to separate domains (assume batch_size * seq_len per domain)
                    total_per_domain = weights.shape[0] // num_domains
                    if total_per_domain * num_domains != weights.shape[0]:
                        continue
                    
                    weights_per_domain = weights.view(num_domains, total_per_domain, -1)
                    
                    # Mean routing per domain: (num_domains, num_experts)
                    domain_means = weights_per_domain.mean(dim=1)
                    
                    # Target: uniform distribution (each domain uses all experts equally)
                    uniform = torch.ones_like(domain_means) / num_experts
                    
                    # KL divergence from uniform - encourages diversity
                    kl_from_uniform = F.kl_div(
                        (domain_means + 1e-8).log(), 
                        uniform, 
                        reduction='batchmean'
                    )
                    
                    # We want routing to be NOT uniform (some specialization)
                    # So we minimize negative KL, but clamp to prevent explosion
                    domain_routing_loss += torch.clamp(-kl_from_uniform, min=-2.0, max=2.0)
                    count += 1
                        
                except Exception as e:
                    logger.debug(f"Could not compute domain routing loss for adapter: {e}")
                    continue
            
            if count > 0:
                domain_routing_loss = domain_routing_loss / count
            
            return domain_routing_loss
            
        except Exception as e:
            logger.error(f"Error in compute_domain_routing_loss: {e}")
            return torch.tensor(0.0, device=x[0].device)
    
    def compute_complexity_invariance_loss(self, adapters):
        """
        Compute complexity invariance loss across experts.
        """
        total_loss = 0.0
        count = 0
        device = None
        
        for m in adapters:
            if device is None and hasattr(m, 'expert_kans'):
                for kan in m.expert_kans:
                    for p in kan.parameters():
                        device = p.device
                        break
                    if device is not None:
                        break
            
            if hasattr(m, 'expert_kans'):
                complexities = []
                for kan in m.expert_kans:
                    # Compute L1 norm of spline weights as complexity measure
                    complexity = 0.0
                    for p in kan.parameters():
                        if p.requires_grad:
                            complexity += p.abs().mean()
                    complexities.append(complexity)
                
                if len(complexities) > 1:
                    # Encourage similar complexity across experts
                    complexities_tensor = torch.stack([c if isinstance(c, torch.Tensor) else torch.tensor(c, device=device) for c in complexities])
                    total_loss += complexities_tensor.var()
                    count += 1
        
        if device is None:
            return torch.tensor(0.0)
        
        if count > 0:
            return total_loss / count
        return torch.tensor(0.0, device=device)
    
    def compute_lipschitz_loss(self, adapters):
        """
        Compute Lipschitz regularization loss for KAN adapters.
        Encourages smooth, stable functions by penalizing large gradients.
        """
        total_loss = 0.0
        count = 0
        device = None
        
        for m in adapters:
            if hasattr(m, 'expert_kans'):
                for kan in m.expert_kans:
                    for name, p in kan.named_parameters():
                        if p.requires_grad:
                            if device is None:
                                device = p.device
                            # Penalize large weight magnitudes (proxy for Lipschitz constant)
                            if 'spline' in name.lower() or 'coef' in name.lower():
                                total_loss += (p ** 2).mean()
                                count += 1
        
        if device is None:
            return torch.tensor(0.0)
        
        if count > 0:
            result = total_loss / count
            if isinstance(result, torch.Tensor):
                return result
            return torch.tensor(result, device=device)
        return torch.tensor(0.0, device=device)
    
    def compute_grid_entropy_loss(self, adapters):
        """
        Compute grid entropy loss to encourage diverse spline activations.
        """
        total_loss = 0.0
        count = 0
        device = None
        
        for m in adapters:
            if hasattr(m, 'expert_kans'):
                for kan in m.expert_kans:
                    if hasattr(kan, 'spline_weight'):
                        w = kan.spline_weight
                        if device is None:
                            device = w.device
                        # Compute entropy of weight distribution
                        w_flat = w.view(-1).abs()
                        w_norm = w_flat / (w_flat.sum() + 1e-8)
                        entropy = -(w_norm * (w_norm + 1e-8).log()).sum()
                        # Maximize entropy (minimize negative entropy)
                        total_loss -= entropy
                        count += 1
        
        if device is None:
            return torch.tensor(0.0)
        
        if count > 0:
            result = total_loss / count
            if isinstance(result, torch.Tensor):
                return result
            return torch.tensor(result, device=device)
        return torch.tensor(0.0, device=device)
    
    def update(self, x, y, **kwargs):
        try:
            all_x = torch.cat([xi for xi in x])
            all_y = torch.cat([yi for yi in y])
            
            # Forward pass
            logits = self.predict(all_x)
            cls_loss = F.cross_entropy(logits, all_y)
            
            # Collect adapters
            from domainbed.models.kan_adapters import MOEBase
            adapters = [m for m in self.network.modules() if isinstance(m, MOEBase)]
            
            # Initialize losses on proper device
            device = all_x.device
            lip_loss = torch.tensor(0.0, device=device)
            ortho_loss = torch.tensor(0.0, device=device)
            load_balance_loss = torch.tensor(0.0, device=device)
            grid_entropy_loss = torch.tensor(0.0, device=device)
            
            # Compute enhanced losses
            for m in adapters:
                m.increment_step()
                
                # Lipschitz regularization
                if hasattr(m, 'compute_lipschitz_loss'):
                    loss_val = m.compute_lipschitz_loss()
                    if loss_val is not None:
                        lip_loss = lip_loss + loss_val.to(device)
                
                # Orthogonality (pairwise or legacy)
                if hasattr(m, 'compute_orthogonality_loss'):
                    loss_val = m.compute_orthogonality_loss()
                    if loss_val is not None:
                        ortho_loss = ortho_loss + loss_val.to(device)
                
                # Load balancing
                if self.enable_load_balance and hasattr(m, 'compute_load_balance_loss'):
                    loss_val = m.compute_load_balance_loss()
                    if loss_val is not None:
                        load_balance_loss = load_balance_loss + loss_val.to(device)
                
                # Grid entropy
                if self.enable_grid_entropy and hasattr(m, 'compute_grid_entropy_loss'):
                    loss_val = m.compute_grid_entropy_loss()
                    if loss_val is not None:
                        grid_entropy_loss = grid_entropy_loss + loss_val.to(device)
            
            # Complexity invariance loss
            comp_loss = self.compute_complexity_invariance_loss(adapters)
            if comp_loss is not None:
                if isinstance(comp_loss, (int, float)):
                    comp_loss = torch.tensor(comp_loss, device=device)
                else:
                    comp_loss = comp_loss.to(device)
            else:
                comp_loss = torch.tensor(0.0, device=device)
            
            # Domain routing loss
            domain_routing_loss = self.compute_domain_routing_loss(adapters, x)
            if domain_routing_loss is not None:
                domain_routing_loss = domain_routing_loss.to(device)
            else:
                domain_routing_loss = torch.tensor(0.0, device=device)
            
            # Total loss
            loss = (cls_loss + 
                   lip_loss * self.lipschitz_weight + 
                   ortho_loss * self.ortho_weight + 
                   comp_loss * self.complexity_weight +
                   load_balance_loss * self.load_balance_weight +
                   grid_entropy_loss * self.grid_entropy_weight +
                   domain_routing_loss * self.domain_routing_weight)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Scale KAN gradients for stability
            for m in adapters:
                if hasattr(m, 'scale_kan_gradients'):
                    m.scale_kan_gradients()
            
            self.optimizer.step()
            
            # Logging
            self.training_step += 1
            if self.training_step % 100 == 0:
                logger.info(f"Step {self.training_step} - Loss: {loss.item():.4f}, "
                           f"Cls: {cls_loss.item():.4f}, Lip: {lip_loss.item():.4f}, "
                           f"Ortho: {ortho_loss.item():.4f}, Comp: {comp_loss.item():.4f}, "
                           f"LoadBal: {load_balance_loss.item():.4f}, "
                           f"GridEnt: {grid_entropy_loss.item():.4f}, "
                           f"DomainRoute: {domain_routing_loss.item():.4f}")
            
            # Track losses
            self.loss_history['total'].append(loss.item())
            self.loss_history['cls'].append(cls_loss.item())
            self.loss_history['lip'].append(lip_loss.item())
            self.loss_history['ortho'].append(ortho_loss.item())
            self.loss_history['comp'].append(comp_loss.item())
            self.loss_history['load_balance'].append(load_balance_loss.item())
            self.loss_history['grid_entropy'].append(grid_entropy_loss.item())
            self.loss_history['domain_routing'].append(domain_routing_loss.item())
            
            return {'loss': loss.item(), 'cls_loss': cls_loss.item()}
            
        except Exception as e:
            logger.error(f"Error in KAN_MOA.update: {e}", exc_info=True)
            # Fallback to simple ERM
            all_x = torch.cat([xi for xi in x])
            all_y = torch.cat([yi for yi in y])
            logits = self.predict(all_x)
            cls_loss = F.cross_entropy(logits, all_y)
            self.optimizer.zero_grad()
            cls_loss.backward()
            self.optimizer.step()
            return {'loss': cls_loss.item(), 'cls_loss': cls_loss.item()}

    def compute_pairwise_orthogonality_loss(self, adapters):
        """
        Compute pairwise orthogonality loss between expert outputs.
        """
        total_loss = 0.0
        count = 0
        device = None
        
        for m in adapters:
            if hasattr(m, 'last_expert_outputs') and m.last_expert_outputs is not None:
                expert_outs = m.last_expert_outputs  # (N, num_experts, r) or similar
                if device is None:
                    device = expert_outs.device
                
                # Ensure 3D tensor for bmm
                while expert_outs.dim() > 3:
                    expert_outs = expert_outs.squeeze(1)
                
                if expert_outs.dim() < 3:
                    continue
                
                N, num_experts, r = expert_outs.shape
                
                if num_experts < 2:
                    continue
                
                # Normalize expert outputs
                expert_outs_norm = F.normalize(expert_outs, dim=-1)  # (N, num_experts, r)
                
                # Compute pairwise cosine similarity
                # (N, num_experts, r) @ (N, r, num_experts) -> (N, num_experts, num_experts)
                similarity = torch.bmm(expert_outs_norm, expert_outs_norm.transpose(1, 2))
                
                # Penalize off-diagonal similarities (want orthogonal experts)
                eye = torch.eye(num_experts, device=device).unsqueeze(0)
                off_diag_sim = (similarity * (1 - eye)).abs()
                ortho_loss = off_diag_sim.mean()
                
                total_loss += ortho_loss
                count += 1
        
        if device is None:
            return torch.tensor(0.0)
        
        if count > 0:
            result = total_loss / count
            if isinstance(result, torch.Tensor):
                return result
            return torch.tensor(result, device=device)
        return torch.tensor(0.0, device=device)
