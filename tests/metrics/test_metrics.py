import os
import random

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn
from torcheval.metrics import BinaryAccuracy, BinaryAUPRC, BinaryAUROC
from torcheval.metrics.functional import binary_accuracy, binary_auprc, binary_auroc

from danling import Metrics, NestedTensor
from danling.metrics import regression_metrics
from danling.metrics.functional import accuracy, auprc, auroc, pearson, rmse, spearman
from danling.metrics.metrics import ScoreMetrics


def demo_dict_metric_func(input, target):
    """
    Normally DanLing Metrics takes a dict of function in constructor calls.
    The function should return a single float value representing the score.

    In rare cases, some metric functions may share internal variables,
    and the best way for them is to use one function to calculate all metrics.

    DanLing Metrics allows functions that return a dict for multiple scores.
    """

    return {"auroc": auroc(input, target), "auprc": auprc(input, target), "acc": accuracy(input, target)}


class Test:
    epsilon = 1e-6

    def test_empty(self):
        metrics = Metrics()
        assert (metrics.inputs == torch.empty(0)).all()
        assert (metrics.targets == torch.empty(0)).all()

    def test_score_metrics(self):
        random.seed(0)
        torch.random.manual_seed(0)
        metrics = ScoreMetrics(auroc=auroc, auprc=auprc, acc=accuracy)
        score_name = "auroc"
        assert metrics.score_name == score_name
        for _ in range(10):
            pred = torch.randn(8).sigmoid()
            target = torch.randint(0, 2, (8,))
            metrics.update(pred, target)
            assert metrics.batch_score == metrics.val[score_name] == metrics.get_score("batch")
            assert metrics.average_score == metrics.avg[score_name] == metrics.get_score("average")
        with pytest.raises(ValueError):
            metrics.get_score("total")
        with pytest.raises(ValueError):
            metrics.score_name = "f1"

    def test_single_tensor_binary(self):
        random.seed(0)
        torch.random.manual_seed(0)
        metrics = Metrics(auroc=auroc, auprc=auprc, acc=accuracy)
        merge_metrics = Metrics(func=demo_dict_metric_func, merge_dict=True)
        auc, prc, acc = BinaryAUROC(), BinaryAUPRC(), BinaryAccuracy()
        preds, targets = [], []
        for _ in range(10):
            pred = torch.randn(8).sigmoid()
            target = torch.randint(0, 2, (8,))
            preds.append(pred)
            targets.append(target)
            metrics.update(pred, target)
            merge_metrics.update(pred, target)
            assert (pred == metrics.input).all()
            assert (target == metrics.target).all()
            auc.update(pred, target)
            prc.update(pred, target)
            acc.update(pred, target)
            assert metrics.compute()["auroc"] - binary_auroc(pred, target) < self.epsilon
            assert metrics.compute()["auprc"] - binary_auprc(pred, target) < self.epsilon
            assert metrics.value()["acc"] - binary_accuracy(pred, target) < self.epsilon
            assert metrics.average()["auroc"] - auc.compute() < self.epsilon
            assert metrics.average()["auprc"] - prc.compute() < self.epsilon
            assert metrics.average()["acc"] - acc.compute() < self.epsilon
            assert metrics.avg == merge_metrics.avg
        assert (torch.cat(preds) == metrics.inputs).all()
        assert (torch.cat(targets) == metrics.targets).all()
        assert metrics.average()["auroc"] - auc.compute() < self.epsilon
        assert metrics.average()["auprc"] - prc.compute() < self.epsilon
        assert metrics.average()["acc"] - acc.compute() < self.epsilon
        assert metrics.avg == merge_metrics.avg

    def test_single_nested_tensor_binary(self):
        random.seed(0)
        torch.random.manual_seed(0)
        metrics = Metrics(auroc=auroc, auprc=auprc, acc=accuracy)
        dict_metrics = Metrics(func=demo_dict_metric_func, merge_dict=False)
        auc, prc, acc = BinaryAUROC(), BinaryAUPRC(), BinaryAccuracy()
        preds, targets = [], []
        lengths_list = [(2, 3, 5, 7), (11, 13, 17, 19)]
        for lengths in lengths_list:
            pred_list, target_list = [], []
            for length in lengths:
                pred_list.append(torch.randn(length).sigmoid())
                target_list.append(torch.randint(0, 2, (length,)))
            preds.extend(pred_list)
            targets.extend(target_list)
            pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
            metrics.update(pred_nt, target_nt)
            dict_metrics.update(pred_nt, target_nt)
            assert (pred_nt == metrics.input).all()
            assert (target_nt == metrics.target).all()
            pred, target = torch.cat(pred_list), torch.cat(target_list)
            auc.update(pred, target)
            prc.update(pred, target)
            acc.update(pred, target)
            assert metrics.compute()["auroc"] - binary_auroc(pred, target) < self.epsilon
            assert metrics.compute()["auprc"] - binary_auprc(pred, target) < self.epsilon
            assert metrics.compute()["acc"] - binary_accuracy(pred, target) < self.epsilon
            assert metrics.average()["auroc"] - auc.compute() < self.epsilon
            assert metrics.average()["auprc"] - prc.compute() < self.epsilon
            assert metrics.average()["acc"] - acc.compute() < self.epsilon
        assert (torch.cat(preds) == metrics.inputs).all()
        assert (torch.cat(targets) == metrics.targets).all()
        assert metrics.average()["auroc"] - auc.compute() < self.epsilon
        assert metrics.average()["auprc"] - prc.compute() < self.epsilon
        assert metrics.average()["acc"] - acc.compute() < self.epsilon
        ret, dict_ret = metrics.average(), dict_metrics.average()
        for key, value in ret.items():
            assert dict_ret[f"func.{key}"] == value

    def test_distributed(self, world_size: int = 8):
        self._test_distributed(self._test_distributed_tensor_binary, world_size)
        self._test_distributed(self._test_distributed_nested_tensor_binary, world_size)
        self._test_distributed(self._test_distributed_nested_tensor_regression, world_size)

    def _test_distributed_tensor_binary(self, rank, world_size):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        metrics = Metrics(auroc=auroc, auprc=auprc, acc=accuracy)
        auc, prc, acc = BinaryAUROC(), BinaryAUPRC(), BinaryAccuracy()
        preds, targets = [], []
        length = 8
        for _ in range(8):
            # PyTorch will complain if requires_grad is not True
            pred = torch.randn(length, requires_grad=True).sigmoid()
            target = torch.randint(0, 2, (length,), dtype=torch.float, requires_grad=True)
            preds.append(pred)
            targets.append(target)
            metrics.update(pred, target)
            auc.update(pred, target)
            prc.update(pred, target)
            acc.update(pred, target)
            assert (pred == metrics.input[length * rank : length * (rank + 1)]).all()  # noqa: E203
            assert (target == metrics.target[length * rank : length * (rank + 1)]).all()  # noqa: E203
            assert metrics.compute()["auroc"] - binary_auroc(pred, target) < self.epsilon
            assert metrics.compute()["auprc"] - binary_auprc(pred, target) < self.epsilon
            assert metrics.compute()["acc"] - binary_accuracy(pred, target) < self.epsilon

        all_preds = torch.cat(self._all_gather(preds, world_size))
        all_targets = torch.cat(self._all_gather(targets, world_size))
        assert (all_preds == metrics.inputs).all()
        assert (all_targets == metrics.targets).all()
        assert metrics.average()["auroc"] - binary_auroc(all_preds, all_targets) < self.epsilon
        assert metrics.average()["auprc"] - binary_auprc(all_preds, all_targets) < self.epsilon
        assert metrics.average()["acc"] - binary_accuracy(all_preds, all_targets) < self.epsilon

        dist.destroy_process_group()

    def _test_distributed_nested_tensor_binary(self, rank, world_size):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # cum_length = 0
        metrics = Metrics(auroc=auroc, auprc=auprc, acc=accuracy)
        auc, prc, acc = BinaryAUROC(), BinaryAUPRC(), BinaryAccuracy()
        preds, targets = [], []
        lengths_list = [[2, 3, 5, 7], [11, 13, 17]]
        if rank == 0:
            lengths_list[-1].append(19)
        # for iter, lengths in enumerate(lengths_list):
        for lengths in lengths_list:
            pred_list, target_list = [], []
            for length in lengths:
                pred_list.append(torch.randn(length).sigmoid())
                target_list.append(torch.randint(0, 2, (length,)))
            preds.extend(pred_list)
            targets.extend(target_list)
            pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
            metrics.update(pred_nt, target_nt)
            pred, target = torch.cat(pred_list), torch.cat(target_list)
            # assert (pred == metrics.input[cum_length * rank : cum_length * (rank + 1)]).all()
            # assert (target == metrics.target[cum_length * rank : cum_length * (rank + 1)]).all()
            assert metrics.compute()["auroc"] - binary_auroc(pred, target) < self.epsilon
            assert metrics.compute()["auprc"] - binary_auprc(pred, target) < self.epsilon
            assert metrics.compute()["acc"] - binary_accuracy(pred, target) < self.epsilon
            pred = torch.cat(self._all_gather(pred_list, world_size))
            target = torch.cat(self._all_gather(target_list, world_size))
            auc.update(pred, target)
            prc.update(pred, target)
            acc.update(pred, target)
            assert metrics.value()["auroc"] - binary_auroc(pred, target) < self.epsilon
            assert metrics.value()["auprc"] - binary_auprc(pred, target) < self.epsilon
            assert metrics.value()["acc"] - binary_accuracy(pred, target) < self.epsilon

        all_preds = torch.cat(self._all_gather(preds, world_size))
        all_targets = torch.cat(self._all_gather(targets, world_size))
        assert (all_preds == metrics.inputs).all()
        assert (all_targets == metrics.targets).all()
        assert metrics.average()["auroc"] - binary_auroc(all_preds, all_targets) < self.epsilon
        assert metrics.average()["auprc"] - binary_auprc(all_preds, all_targets) < self.epsilon
        assert metrics.average()["acc"] - binary_accuracy(all_preds, all_targets) < self.epsilon

        dist.destroy_process_group()

    def _test_distributed_nested_tensor_regression(self, rank, world_size):
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        # cum_length = 0
        metrics = regression_metrics()
        preds, targets = [], []
        lengths_list = [[2, 3, 5, 7], [11, 13, 17]]
        if rank == 0:
            lengths_list[-1].append(19)
        # for iter, lengths in enumerate(lengths_list):
        for lengths in lengths_list:
            pred_list, target_list = [], []
            for length in lengths:
                pred_list.append(torch.randn(length))
                target_list.append(torch.randn(length))
            preds.extend(pred_list)
            targets.extend(target_list)
            pred_nt, target_nt = NestedTensor(pred_list), NestedTensor(target_list)
            metrics.update(pred_nt, target_nt)
            pred, target = torch.cat(pred_list), torch.cat(target_list)
            assert metrics.compute()["pearson"] - pearson(pred, target) < self.epsilon
            assert metrics.compute()["spearman"] - spearman(pred, target) < self.epsilon
            assert metrics.compute()["rmse"] - rmse(pred, target) < self.epsilon
            pred = torch.cat(self._all_gather(pred_list, world_size))
            target = torch.cat(self._all_gather(target_list, world_size))
            assert metrics.value()["pearson"] - pearson(pred, target) < self.epsilon
            assert metrics.value()["spearman"] - spearman(pred, target) < self.epsilon
            assert metrics.value()["rmse"] - rmse(pred, target) < self.epsilon

        all_preds = torch.cat(self._all_gather(preds, world_size))
        all_targets = torch.cat(self._all_gather(targets, world_size))
        assert (all_preds == metrics.inputs).all()
        assert (all_targets == metrics.targets).all()
        assert metrics.average()["pearson"] - pearson(all_preds, all_targets) < self.epsilon
        assert metrics.average()["spearman"] - spearman(all_preds, all_targets) < self.epsilon
        assert metrics.average()["rmse"] - rmse(all_preds, all_targets) < self.epsilon

        dist.destroy_process_group()

    def _test_distributed(self, func: callable, world_size: int = 8):
        random.seed(0)
        torch.random.manual_seed(0)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(29501)
        spawn(func, args=(world_size,), nprocs=world_size)

    def _gather_tensors(self, tensor, world_size):
        tensor = torch.cat(tensor)
        synced_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather_object(synced_tensor, tensor)
        return synced_tensor

    def _all_gather(self, tensors, world_size):
        synced_tensor = [None for _ in range(world_size)]
        dist.all_gather_object(synced_tensor, tensors)
        return [i for t in synced_tensor for i in t]
