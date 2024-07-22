import pickle

from torch import ge
from fastNLP.core.dataset import DataSet
from fastNLP.core.batch import DataSetIter
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.embeddings import StaticEmbedding
from fastNLP.core.sampler import BucketSampler, ConstantTokenNumSampler
from torch.utils.data import Sampler
from collections import defaultdict
import os
import random


class DataModule:
    def __init__(
        self, hparams, generator=None, worker_init_fn=None, mask="<mask>"
    ):
        super().__init__()

        self.hparams = hparams
        self.device = self.hparams.device
        self.generator = generator
        self.worker_init_fn = worker_init_fn
        self.mask = mask
        self.setup()

    def prepare_data(self):
        pass

    def setup(self):
        data = self.hparams.data
        train_dataset = DataSet()
        val_dataset = DataSet()
        test_dataset = DataSet()
        word_vocab = Vocabulary(max_size=data.vocab_size, mask=self.mask)
        train_data = pickle.load(open(data.train_file, "rb"))
        val_data = pickle.load(open(data.val_file, "rb"))
        test_data = pickle.load(open(data.test_file, "rb"))

        input_fields = []
        target_fields = []

        train_dataset.add_field(
            "sentence", train_data["word"], ignore_type=True
        )
        val_dataset.add_field("sentence", val_data["word"], ignore_type=True)
        test_dataset.add_field("sentence", test_data["word"], ignore_type=True)

        target_fields.append("sentence")

        # only for lexicalized PCFGs.
        try:
            val_dataset.add_field(
                "head", val_data["head"], padder=None, ignore_type=True
            )
            test_dataset.add_field(
                "head", test_data["head"], padder=None, ignore_type=True
            )
            val_dataset.set_target("head")
            test_dataset.set_target("head")
        except:
            print("No head")
            pass

        train_dataset.add_field(
            "gold_tree", train_data["gold_tree"], padder=None, ignore_type=True
        )
        val_dataset.add_field(
            "gold_tree", val_data["gold_tree"], padder=None, ignore_type=True
        )
        test_dataset.add_field(
            "gold_tree", test_data["gold_tree"], padder=None, ignore_type=True
        )
        # To eval train dataset
        target_fields.append("gold_tree")

        # train_dataset.add_seq_len(field_name="word", new_field_name="seq_len")
        # val_dataset.add_seq_len(field_name="word", new_field_name="seq_len")
        # test_dataset.add_seq_len(field_name="word", new_field_name="seq_len")
        train_dataset.add_seq_len(
            field_name="sentence", new_field_name="seq_len"
        )
        val_dataset.add_seq_len(
            field_name="sentence", new_field_name="seq_len"
        )
        test_dataset.add_seq_len(
            field_name="sentence", new_field_name="seq_len"
        )

        # Binarized gold trees
        try:
            train_dataset.add_field(
                "gold_tree_left",
                train_data["gold_tree_left"],
                padder=None,
                ignore_type=True,
            )
            val_dataset.add_field(
                "gold_tree_left",
                val_data["gold_tree_left"],
                padder=None,
                ignore_type=True,
            )
            test_dataset.add_field(
                "gold_tree_left",
                test_data["gold_tree_left"],
                padder=None,
                ignore_type=True,
            )
            target_fields.append("gold_tree_left")
        except:
            print("No Left Binarization")
            pass

        try:
            train_dataset.add_field(
                "gold_tree_right",
                train_data["gold_tree_right"],
                padder=None,
                ignore_type=True,
            )
            val_dataset.add_field(
                "gold_tree_right",
                val_data["gold_tree_right"],
                padder=None,
                ignore_type=True,
            )
            test_dataset.add_field(
                "gold_tree_right",
                test_data["gold_tree_right"],
                padder=None,
                ignore_type=True,
            )
            target_fields.append("gold_tree_right")
        except:
            print("No Right Binarization")
            pass

        # Depth of trees
        try:
            train_dataset.add_field("depth", train_data["depth"], padder=None)
            val_dataset.add_field("depth", val_data["depth"], padder=None)
            test_dataset.add_field("depth", test_data["depth"], padder=None)
            target_fields.append("depth")
        except:
            print("No depth")
            pass

        try:
            train_dataset.add_field(
                "depth_left", train_data["depth_left"], padder=None
            )
            val_dataset.add_field(
                "depth_left", val_data["depth_left"], padder=None
            )
            test_dataset.add_field(
                "depth_left", test_data["depth_left"], padder=None
            )
            target_fields.append("depth_left")
        except:
            print("No depth of left binarization")
            pass

        try:
            train_dataset.add_field(
                "depth_right", train_data["depth_right"], padder=None
            )
            val_dataset.add_field(
                "depth_right", val_data["depth_right"], padder=None
            )
            test_dataset.add_field(
                "depth_right", test_data["depth_right"], padder=None
            )
            target_fields.append("depth_right")
        except:
            print("No depth of right binarization")
            pass

        # POS tag of gold trees
        try:
            train_dataset.add_field(
                "pos", train_data["pos"], padder=None, ignore_type=True
            )
            val_dataset.add_field(
                "pos", val_data["pos"], padder=None, ignore_type=True
            )
            test_dataset.add_field(
                "pos", test_data["pos"], padder=None, ignore_type=True
            )
            target_fields.append("pos")
        except:
            print("No pos")
            pass

        def clean_word(words):
            import re

            def clean_number(w):
                new_w = re.sub("[0-9]{1,}([,.]?[0-9]*)*", "N", w)
                return new_w

            return [clean_number(word.lower()) for word in words]

        train_dataset.apply_field(clean_word, "sentence", "word")
        val_dataset.apply_field(clean_word, "sentence", "word")
        test_dataset.apply_field(clean_word, "sentence", "word")

        word_vocab.from_dataset(train_dataset, field_name="word")
        word_vocab.index_dataset(train_dataset, field_name="word")
        word_vocab.index_dataset(val_dataset, field_name="word")
        word_vocab.index_dataset(test_dataset, field_name="word")

        # drop length 1 sentences. As S->NT, while NT cannot generate single word in our
        # settings (only preterminals generate words
        self.val_dataset = val_dataset.drop(
            lambda x: x["seq_len"] == 1, inplace=True
        )
        self.train_dataset = train_dataset.drop(
            lambda x: x["seq_len"] == 1, inplace=True
        )
        self.test_dataset = test_dataset.drop(
            lambda x: x["seq_len"] == 1, inplace=True
        )

        self.word_vocab = word_vocab
        input_fields.append("seq_len")
        input_fields.append("word")
        self.train_dataset.set_input(*input_fields)
        self.val_dataset.set_input(*input_fields)
        self.test_dataset.set_input(*input_fields)

        self.train_dataset.set_target(*target_fields)
        self.val_dataset.set_target(*target_fields)
        self.test_dataset.set_target(*target_fields)
        # For L-PCFGs.

    def train_dataloader(self, max_len=40, min_len=0):
        args = self.hparams.train
        train_dataset = self.train_dataset.drop(
            lambda x: x["seq_len"] < min_len, inplace=False
        )
        train_dataset = train_dataset.drop(
            lambda x: x["seq_len"] > max_len, inplace=False
        )
        train_sampler = ByLengthSampler(
            dataset=train_dataset, batch_size=args.batch_size
        )
        return DataSetIter(
            dataset=train_dataset,
            batch_sampler=train_sampler,
            generator=self.generator,
            worker_init_fn=self.worker_init_fn,
        )

    # @property
    def val_dataloader(self, max_len=None):
        args = self.hparams.test
        if max_len is not None:
            val_dataset = self.val_dataset.drop(
                lambda x: x["seq_len"] > max_len, inplace=False
            )
        else:
            val_dataset = self.val_dataset
        if args.sampler == "token":
            test_sampler = ConstantTokenNumSampler(
                seq_len=val_dataset.get_field("seq_len").content,
                max_token=args.max_tokens,
                num_bucket=args.bucket,
            )
            return DataSetIter(
                val_dataset,
                batch_size=1,
                sampler=None,
                as_numpy=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
                timeout=0,
                worker_init_fn=self.worker_init_fn,
                batch_sampler=test_sampler,
                generator=self.generator,
            )
        elif args.sampler == "batch":
            train_sampler = ByLengthSampler(
                dataset=val_dataset, batch_size=args.batch_size
            )
            return DataSetIter(
                dataset=val_dataset,
                batch_sampler=train_sampler,
                generator=self.generator,
                worker_init_fn=self.worker_init_fn,
            )
        else:
            raise NotImplementedError

    @property
    def test_dataloader(self):
        args = self.hparams.test
        test_dataset = self.test_dataset
        if args.sampler == "token":
            test_sampler = ConstantTokenNumSampler(
                seq_len=test_dataset.get_field("seq_len").content,
                max_token=args.max_tokens,
                num_bucket=args.bucket,
            )
            return DataSetIter(
                self.test_dataset,
                batch_size=1,
                sampler=None,
                as_numpy=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
                timeout=0,
                worker_init_fn=self.worker_init_fn,
                batch_sampler=test_sampler,
                generator=self.generator,
            )
        elif args.sampler == "batch":
            train_sampler = ByLengthSampler(
                dataset=test_dataset, batch_size=args.batch_size
            )
            return DataSetIter(
                dataset=test_dataset,
                batch_sampler=train_sampler,
                generator=self.generator,
                worker_init_fn=self.worker_init_fn,
            )
        else:
            raise NotImplementedError


"""
Same as (Kim et al, 2019)
"""


class ByLengthSampler(Sampler):
    def __init__(self, dataset, batch_size=4):
        self.group = defaultdict(list)
        self.seq_lens = dataset["seq_len"]
        for i, length in enumerate(self.seq_lens):
            self.group[length].append(i)
        self.batch_size = batch_size
        total = []

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        for idx, lst in self.group.items():
            total = total + list(chunks(lst, self.batch_size))
        self.total = total

    def __iter__(self):
        random.shuffle(self.total)
        for batch in self.total:
            yield batch

    def __len__(self):
        return len(self.total)
