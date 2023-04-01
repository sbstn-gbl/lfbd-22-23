import collections
import itertools
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch


def baskets_df_to_list(
    x: pd.DataFrame,
    min_basket_size: int = 0,
    shuffle: bool = True,
    to_string: bool = True,
    seed: int = 123,
) -> list:

    # create raw basket list
    x_basket_product = x[["basket", "product"]].sort_values("basket")
    x_basket_product["n"] = x_basket_product.groupby("basket").product.transform(
        "count"
    )
    x_basket_product = x_basket_product[x_basket_product["n"] >= min_basket_size]
    del x_basket_product["n"]
    keys, values = x_basket_product.values.T
    ukeys, index = np.unique(keys, True)
    values = values.astype(str)
    arrays = np.split(values, index)
    basket_list = [list(a) for a in arrays[1:]]

    # randomise basket order and product order in baskets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(basket_list)
        for i in range(len(basket_list)):
            np.random.shuffle(basket_list[i])

    # return
    return basket_list


def gensim_to_pandas(model):
    vocab = []
    for key, value in model.wv.key_to_index.items():
        vocab.append(
            pd.DataFrame(
                {
                    "product": [int(key)],
                    "index": value,
                    "N": model.wv.get_vecattr(key, "count"),
                }
            )
        )
    vocab = pd.concat(vocab).set_index("index").sort_index()
    syn0 = pd.DataFrame(model.wv.vectors, index=vocab["product"]).sort_index()
    syn0.columns = [str(x) for x in syn0.columns]
    syn1 = pd.DataFrame(model.syn1neg, index=vocab["product"]).sort_index()
    syn1.columns = [str(x) for x in syn1.columns]
    return syn0, syn1, vocab


def plot_tsne(data, plotlabels=False, scatter=0):

    x_value = data["x"].values
    y_value = data["y"].values
    category_label = data["category"].values

    if scatter > 0:
        x_value = x_value + np.random.uniform(-scatter, scatter, len(x_value))
        y_value = y_value + np.random.uniform(-scatter, scatter, len(y_value))

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.tight_layout()
    plt.scatter(x_value, y_value, c=category_label, alpha=0.1, s=500)

    if plotlabels:
        for x, y, c in zip(x_value, y_value, category_label):
            ax.text(
                x,
                y,
                c,
                size=10,
                horizontalalignment="center",
                verticalalignment="center",
            )

    _ = plt.xlim(x_value.min() * 1.05, x_value.max() * 1.05)
    _ = plt.ylim(y_value.min() * 1.05, y_value.max() * 1.05)


class NegativeSamplesGenerator:
    """
    Class for generating negative samples
        get_negative_samples: produce negative samples
    """

    def __init__(
        self,
        data: pd.DataFrame,
        n_negative_samples: int,
        batch_size: int,
        power: float = 0.75,
        variable_product: str = "product",
    ) -> None:
        """
        Initialize negative samples generator (for given positive samples)
            data: basket data, must contain `variable_basket` and `variable_product`
            n_negative_samples: number of negative samples per positive sample
            batch_size: size of a single batch
            power: distortion factor for negative sample generator
            variable_product: product identifier in `data`
        """

        self.n_negative_samples = n_negative_samples
        self.batch_size = batch_size
        self.power = power
        self.n_draws = self.batch_size * self.n_negative_samples
        self.variable_product = variable_product
        self.domain = (2 ** 31 - 1,)
        self._build_product_counts(data)
        self._build_cumulative_count_table()
        self.products = np.array(list(self.counts.keys()))

    def get_negative_samples(self, context: np.ndarray = None) -> np.ndarray:
        """
        Produce negative samples (for given positive samples)
            context: context products that may not be used as negative samples
        """
        if context is not None:
            negative_samples = (
                np.zeros((self.n_negative_samples, len(context)), dtype=np.int32) - 1
            )
            done_sampling = False
            while not done_sampling:
                new_sample_index = negative_samples == -1
                n_draws = np.sum(new_sample_index)
                random_integers = np.random.randint(0, self.domain, n_draws)
                new_negative_samples_index = np.searchsorted(
                    self.cumulative_count_table, random_integers
                )
                new_negative_samples = self.products[new_negative_samples_index]
                negative_samples[new_sample_index] = new_negative_samples
                negative_samples[negative_samples == context] = -1
                done_sampling = np.all(negative_samples != -1)
            return negative_samples
        else:
            random_integers = np.random.randint(0, self.domain, self.n_draws)
            negative_samples_index = np.searchsorted(
                self.cumulative_count_table, random_integers
            )
            return self.products[negative_samples_index].reshape(
                (self.batch_size, self.n_negative_samples)
            )

    def _build_product_counts(self, x: pd.DataFrame) -> None:
        """
        Count number of times products occur in basket data
        """
        n_products = x[self.variable_product].max() + 1
        product_counts = (
            x.groupby(self.variable_product)[self.variable_product].count().to_dict()
        )
        product_counts_filled = collections.OrderedDict()
        for j in range(n_products):
            if j not in product_counts:
                product_counts_filled[j] = 0
            else:
                product_counts_filled[j] = product_counts[j]
        self.counts = product_counts_filled

    def _build_cumulative_count_table(self) -> None:
        """
        Build count table (mapped to self.domain) for integer sampling of products
        """
        tmp = np.array(list(self.counts.values())) ** self.power
        cumulative_relative_count_table = np.cumsum(tmp / sum(tmp))
        self.cumulative_count_table = np.int32(
            (cumulative_relative_count_table * self.domain).round()
        )
        assert self.cumulative_count_table[-1] == self.domain


class DataStreamP2V:
    """
    Class for generating P2V training samples
        generate_batch: produce a batch of training samples
        reset_iterator: reset data streamer and empty sample cache
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variable_basket: str,
        variable_product: str,
        batch_size: int = 8_192,
        shuffle: bool = True,
        n_negative_samples: int = 0,
        power: float = 0.75,
        allow_context_collisions: bool = False,
    ):
        """
        Initialize P2V data streamer
            data: must contain `variable_basket` and `variable_product`
            variable_basket: basket identifier in `data`
            variable_product: product identifier in `data`
            batch_size: size of a single batch
            shuffle: shuffle data when resetting streamer
            n_negative_samples: number of negative samples per positive sample
            power: distortion factor for negative sample generator
            allow_context_collisions: allow that an id is a positive and a negative sample at the same time
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cached_samples = []
        self.basket_list = self._basket_df_to_list(
            x=data, variable_basket=variable_basket, variable_product=variable_product
        )
        self.reset_iterator()
        self.produce_negative_samples = n_negative_samples > 0
        if self.produce_negative_samples:
            self.allow_context_collisions = allow_context_collisions
            self.negative_samples_generator = NegativeSamplesGenerator(
                data=data,
                n_negative_samples=n_negative_samples,
                batch_size=self.batch_size,
                power=power,
                variable_product=variable_product,
            )

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce a batch of training samples containing center products, context products, and negative samples
        """
        # fill cache
        self._fill_cache()

        # generate skip-gram pairs
        output_array = np.asarray(self.cached_samples[: self.batch_size])
        self.cached_samples = self.cached_samples[self.batch_size :]
        center = output_array[:, 0, 0].astype(np.int64)
        context = output_array[:, 1, 0].astype(np.int64)

        # add negative samples
        if self.produce_negative_samples:
            if self.allow_context_collisions:
                negative_samples = (
                    self.negative_samples_generator.get_negative_samples()
                )
            else:
                negative_samples = self.negative_samples_generator.get_negative_samples(
                    context
                ).T
        else:
            negative_samples = np.empty(shape=(2, 0))

        # return
        return center, context, negative_samples

    def reset_iterator(self) -> None:
        """
        Reset data streamer and empty sample cache
        """
        if self.shuffle:
            np.random.shuffle(self.basket_list)
        self.basket_iterator = self._basket_iterator(self.basket_list)
        self.cached_samples = []

    def _basket_df_to_list(self, x, variable_basket, variable_product):
        """
        Turn a basket dataframe into a list of baskets
        """
        x_basket_values = (
            x[[variable_basket, variable_product]].sort_values([variable_basket]).values
        )
        keys = x_basket_values[:, 0]
        ukeys, index = np.unique(keys, True)
        return np.split(x_basket_values[:, 1:], index)[1:]

    def _basket_iterator(self, basket_list):
        """
        Iterator yielding single baskets
        """
        for basket in basket_list:
            yield basket

    def _fill_cache(self):
        """
        Fill sample cache with center-context pairs
        """
        fill_cache = len(self.cached_samples) < self.batch_size
        while fill_cache:
            try:
                new_basket = next(self.basket_iterator, None)
                self.cached_samples.extend(itertools.permutations(new_basket, 2))
            except:
                fill_cache = False
            if len(self.cached_samples) >= self.batch_size:
                fill_cache = False


def build_data_loader(streamer, config):
    # build numpy arrays for full dataset
    streamer.reset_iterator()
    list_ce = []
    list_co = []
    list_ns = []
    while True:
        try:
            ce, co, ns = streamer.generate_batch()
            list_ce.append(ce)
            list_co.append(co)
            list_ns.append(ns)
        except:
            break
    ce = np.hstack(list_ce)
    co = np.hstack(list_co)
    ns = np.vstack(list_ns)

    # train/validation split
    ce_t, ce_v, co_t, co_v, ns_t, ns_v = sklearn.model_selection.train_test_split(
        ce, co, ns, **config["split"]
    )

    # build data loader
    dl_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.LongTensor(ce_t),
            torch.LongTensor(co_t),
            torch.LongTensor(ns_t),
        ),
        **config["train"],
    )
    dl_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.LongTensor(ce_v),
            torch.LongTensor(co_v),
            torch.LongTensor(ns_v),
        ),
        **config["valid"],
    )
    return dl_train, dl_val
