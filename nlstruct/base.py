import time
from collections import defaultdict

import pyarrow as pa
import pytorch_lightning as pl
import torch
from transformers.trainer_pt_utils import LengthGroupedSampler


class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, epoch_length=None, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        print("epoch_length", epoch_length)
        self.epoch_length = epoch_length if epoch_length is not None else len(dataset)
        self._iterator = None

    def __len__(self):
        return self.epoch_length

    def __iter__(self):
        if self._iterator is None:
            self._iterator = self._get_iterator()
        for res in range(self.epoch_length):
            try:
                yield next(self._iterator)
            except StopIteration:
                self._iterator = self._get_iterator()
                yield next(self._iterator)


def make_grouped_sampler(dataset, column, batch_size, seed=42, distributed_sampler_kwargs={}):
    if isinstance(dataset, datasets.Dataset):
        if isinstance(dataset.features[column], datasets.Sequence):
            lengths = pa.compute.list_value_length(dataset.data.table[column]).to_pylist()
        else:
            lengths = dataset.data.table[column].to_pandas().to_list()
    elif isinstance(dataset, list):
        lengths = [doc[column] for doc in dataset]
        if not isinstance(lengths[0], (int, float)):
            lengths = [len(l) for l in lengths]
    if distributed_sampler_kwargs.get("num_replicas", 1) <= 1:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return LengthGroupedSampler(
            dataset=dataset,
            batch_size=batch_size,
            lengths=lengths,
            model_input_name=None,
            generator=generator,
        )
    else:
        return DistributedLengthGroupedSampler(
            dataset=dataset,
            batch_size=batch_size,
            lengths=lengths,
            model_input_name=None,
            seed=seed,
            **distributed_sampler_kwargs,
        )


def identity(x): return x


def make_dataloader(dataset, batching, epoch_length=None, distributed_sampler_kwargs=None):
    if distributed_sampler_kwargs is None:
        distributed_sampler_kwargs = {}
    if dataset is None:
        return None
    if isinstance(dataset, dict):
        return {name: (make_dataloader(dataset[name], batching[name], epoch_length=epoch_length))
                for name in dataset if dataset[name] is not None}
    batching = dict(batching)
    group_by_length = batching.pop("group_by_length", None)
    if group_by_length:
        batch_size = batching.get("batch_size")
        sampler = make_grouped_sampler(dataset, group_by_length, batch_size, distributed_sampler_kwargs=distributed_sampler_kwargs)
        batching["sampler"] = sampler
        batching["batch_sampler"] = None
    if epoch_length is not None:
        return CustomDataLoader(dataset=dataset, epoch_length=epoch_length, **batching, collate_fn=identity)
    else:
        return torch.utils.data.DataLoader(dataset=dataset, **batching, collate_fn=identity)


class BaseModule(pl.LightningModule):
    def __init__(self, batching, metrics=None, gradient_clip_val=None):
        super().__init__()
        self.metrics = torchmetrics.MetricCollection({k: get_instance(m) for k, m in metrics.items()} if metrics is not None else {})

        train_batching = batching.get("train", batching)
        eval_batching = batching.get("eval", train_batching)
        self.batching = {"train": train_batching, "eval": eval_batching}
        self._time = 0

    def transfer_batch_to_device(self, inputs, device):
        return inputs

    @property
    def train_dataloader(self):
        def fn(): return make_dataloader(
            self.preprocessor.preprocess(self.train_data),
            self.batching['train'],
            epoch_length=self.trainer.val_check_interval if self.trainer is not None else 2000,
            distributed_sampler_kwargs=self.trainer.distributed_sampler_kwargs if self.trainer is not None else {},
        )

        return fn

    @property
    def val_dataloader(self):
        def fn(): return make_dataloader(
            self.preprocessor.preprocess(self.val_data),
            self.batching['eval'],
            distributed_sampler_kwargs=self.trainer.distributed_sampler_kwargs,
        ) if self.val_data is not None else None

        return fn

    @property
    def test_dataloader(self):
        def fn(): return make_dataloader(
            self.preprocessor.preprocess(self.test_data),
            self.batching['eval'],
            distributed_sampler_kwargs=self.trainer.distributed_sampler_kwargs,
        ) if self.test_data is not None else None

        return fn

    @train_dataloader.setter
    def train_dataloader(self, data):
        self.train_data = data()
        if hasattr(self.train_data, 'dataset'):
            self.train_data = self.train_data.dataset

    @val_dataloader.setter
    def val_dataloader(self, data):
        self.val_data = data()
        if hasattr(self.val_data, 'dataset'):
            self.val_data = self.val_data.dataset

    @test_dataloader.setter
    def test_dataloader(self, data):
        self.test_data = data()
        if hasattr(self.test_data, 'dataset'):
            self.test_data = self.test_data.dataset

    def split_mini_batch(self, samples):
        return [samples]

    def validation_step(self, samples, batch_idx):
        return self.predict_step(samples)

    def validation_epoch_end(self, outputs):
        results = self.preprocessor.postprocess_dataset(
            list(chain.from_iterable(outputs)),
            self.val_data,
        )
        self.log_dict({
            ("val_{}_{}".format(name, field) if field else "val_{}".format(name, )): value
            for name, metric in self.metrics(results, self.val_data).items()
            for field, value in metric.items()
        })

    def on_validation_epoch_start(self):
        self.metrics.reset()

    test_step = validation_step
    on_test_epoch_start = on_validation_epoch_start

    def on_train_epoch_start(self):
        self._time = time.time()

    def test_epoch_end(self, outputs):
        print("outputs", len(outputs))
        results = self.preprocessor.postprocess_dataset(
            list(chain.from_iterable(outputs)),
            self.test_data,
        )
        print("results", len(results))
        self.log_dict({
            ("test_{}_{}".format(name, field) if field else "test_{}".format(name, )): value
            for name, metric in self.metrics(results, self.test_data).items()
            for field, value in metric.items()
        })

    def training_step(self, samples, batch_idx):
        opt = self.optimizers()

        opt.zero_grad()

        total_loss = torch.zeros((), device=self.device)
        losses_and_counts = defaultdict(lambda: torch.zeros((), device=self.device))
        for mini_batch in self.split_mini_batch(samples):
            res = self(mini_batch)
            loss = res["loss"]
            total_loss += loss.detach()
            for key, value in res.items():
                if key.endswith("loss"):
                    losses_and_counts[key] += value.detach() if hasattr(value, 'detach') else value
                elif key.endswith("count"):
                    losses_and_counts[key] += value
            self.manual_backward(loss / 100.)
            del loss, res, key, value

        torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)

        opt.step()

        return {"loss": total_loss, **losses_and_counts}

    def on_train_epoch_start(self):
        self._time = time.time()

    def training_epoch_end(self, outputs):
        totals = defaultdict(lambda: 0)
        for output in outputs:
            for key, value in output.items():
                if key.endswith("count"):
                    totals[key] += value
        max_grad = max(output["max_grad"] for output in outputs) if "max_grad" in outputs[0] else None
        for key in outputs[0].keys():
            if key.endswith("loss"):
                count_key = key.replace("loss", "count")
                self.log(key, sum(float(output[key]) * output.get(count_key, 1) for output in outputs) / totals.get(count_key, len(outputs)))
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        if max_grad is not None:
            self.log("max_grad", max_grad)

    def on_epoch_end(self):
        self.log("duration", time.time() - self._time)

    def predict(self, data):
        if isinstance(data, datasets.Dataset):
            return data.map(lambda samples: to_dol(self.predict(to_lod(samples))),
                            batched=True,
                            remove_columns=data.column_names)

        elif isinstance(data, list):
            preprocessed = self.preprocessor.preprocess(data, with_supervision=False)
            dl = make_dataloader(
                preprocessed,
                self.batching['eval'],
                distributed_sampler_kwargs={},
            )
            results = self.preprocessor.postprocess_dataset(
                list(chain.from_iterable(self.predict_step(samples) for samples in dl)),
                data,
            )
            return results

        raise Exception()


def to_dol(x):
    dico = {key: [] for key in x[0]}
    for key in dico:
        dico[key] = [item[key] for item in x]
    return dico


def to_lod(x):
    keys, values = zip(*x.items())
    return [dict(zip(keys, vals)) for vals in zip(*values)]


import datasets


def merge_datasets(**kwargs):
    return datasets.DatasetDict({
        'train': datasets.DatasetDict({name: ds['train'] for name, ds in kwargs.items() if 'train' in ds}),
        'validation': datasets.DatasetDict({name: ds['validation'] for name, ds in kwargs.items() if 'validation' in ds}),
        'test': datasets.DatasetDict({name: ds['test'] for name, ds in kwargs.items() if 'test' in ds}),
    })


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, data):
        super().__init__()
        if isinstance(data, dict):
            self._train_dataloader = data.get('train', None)
            self._val_dataloader = data.get('validation', None)
            self._test_dataloader = data.get('test', None)
        else:
            self._train_dataloader = data
            self._val_dataloader = None
            self._test_dataloader = None

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader
