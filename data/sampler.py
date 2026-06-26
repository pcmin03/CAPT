import numpy as np
import random
import torch
from torch.utils.data import sampler


# Here we define a Sampler that has all the samples of each batch from the same domain,
# same as before but not distributed

# gives equal number of samples per domain, ordered across domains

class BalancedSampler(sampler.Sampler):

    def __init__(self, domain_ids, samples_per_domain, domains_per_batch=5, iters='min'):

        self.n_doms = domains_per_batch
        self.domain_ids = domain_ids
        random.seed(0)

        self.dict_domains = {}
        self.indeces = {}

        for i in range(self.n_doms):
            self.dict_domains[i] = []
            self.indeces[i] = 0

        self.dpb = domains_per_batch
        self.dbs = samples_per_domain
        self.bs = self.dpb*self.dbs

        for idx, d in enumerate(self.domain_ids):
            self.dict_domains[d].append(idx)

        min_dom = 10000000
        max_dom = 0

        for d in self.domain_ids:
            if len(self.dict_domains[d]) < min_dom:
                min_dom = len(self.dict_domains[d])
            if len(self.dict_domains[d]) > max_dom:
                max_dom = len(self.dict_domains[d])

        if iters == 'min':
            self.iters = min_dom // self.dbs
        elif iters == 'max':
            self.iters = max_dom // self.dbs
        else:
            self.iters = int(iters)

        for idx in range(self.n_doms):
            random.shuffle(self.dict_domains[idx])


    def _sampling(self, d_idx, n):
        if self.indeces[d_idx] + n >= len(self.dict_domains[d_idx]):
            self.dict_domains[d_idx] += self.dict_domains[d_idx]
        self.indeces[d_idx] = self.indeces[d_idx] + n
        return self.dict_domains[d_idx][self.indeces[d_idx] - n:self.indeces[d_idx]]


    def _shuffle(self):
        sIdx = []
        for i in range(self.iters):
            for j in range(self.n_doms):
                sIdx += self._sampling(j, self.dbs)
        return np.array(sIdx)


    def __iter__(self):
        return iter(self._shuffle())


    def __len__(self):
        return self.iters * self.bs


class PKDomainSampler(sampler.Sampler):
    """PK-style class-balanced sampler for UCDR (N3 from BREAKTHROUGH_STRATEGY).

    Each batch contains P classes × K samples per class. K samples are
    distributed across domains when possible (cross-domain enforcement).

    Default: P=12 × K=5 = 60 (same batch size as BalancedSampler).
    Cross-domain positives per batch: 12 × 4 (or more) = 48+ same-class
    cross-domain pairs (vs ~7-8 in BalancedSampler).

    Args:
        domain_ids: (N,) numeric domain assignment for each sample
        class_ids: (N,) numeric class assignment for each sample
        P: classes per batch (default 12)
        K: samples per class (default 5)
        n_domains_train: number of training domains (for cross-domain spread)
        iters: 'min', 'max', or int — sampler length controller
    """

    def __init__(self, domain_ids, class_ids, P=12, K=5,
                 n_domains_train=None, iters='min', seed=0):
        random.seed(seed)
        np.random.seed(seed)

        self.P = P
        self.K = K
        self.bs = P * K

        self.domain_ids = np.array(domain_ids)
        self.class_ids = np.array(class_ids)

        # Build (class, domain) → list of indices
        self.class_to_dom_idx = {}  # cls → {dom: [indices]}
        for idx, (c, d) in enumerate(zip(self.class_ids, self.domain_ids)):
            c_int, d_int = int(c), int(d)
            if c_int not in self.class_to_dom_idx:
                self.class_to_dom_idx[c_int] = {}
            if d_int not in self.class_to_dom_idx[c_int]:
                self.class_to_dom_idx[c_int][d_int] = []
            self.class_to_dom_idx[c_int][d_int].append(idx)

        self.all_classes = sorted(self.class_to_dom_idx.keys())
        self.n_classes = len(self.all_classes)
        self.n_domains_train = (n_domains_train
                                or len(set(int(d) for d in self.domain_ids)))

        # Compute iters by total samples ÷ batch_size
        total_samples = len(self.domain_ids)
        if iters == 'min':
            self.iters = total_samples // self.bs
        elif iters == 'max':
            self.iters = total_samples // self.bs
        else:
            self.iters = int(iters)

        # Per-class shuffle state
        self._cls_dom_iter = {}
        self._init_class_iters()

    def _init_class_iters(self):
        """For each class, prepare a randomized order of (dom, idx) pairs."""
        for c in self.all_classes:
            dom_idx = self.class_to_dom_idx[c]
            # Round-robin across domains for cross-domain coverage
            domains = sorted(dom_idx.keys())
            random.shuffle(domains)
            # Per-domain shuffle
            for d in domains:
                random.shuffle(dom_idx[d])
            self._cls_dom_iter[c] = {
                "dom_order": domains,
                "dom_ptrs": {d: 0 for d in domains},
                "next_dom_idx": 0,
            }

    def _sample_class(self, c):
        """Sample K indices for class c, spread across domains via round-robin."""
        state = self._cls_dom_iter[c]
        dom_idx = self.class_to_dom_idx[c]
        domains = state["dom_order"]
        if not domains:
            return []
        sampled = []
        attempts = 0
        max_attempts = self.K * 4  # safety
        while len(sampled) < self.K and attempts < max_attempts:
            d = domains[state["next_dom_idx"] % len(domains)]
            state["next_dom_idx"] += 1
            ptr = state["dom_ptrs"][d]
            if ptr >= len(dom_idx[d]):
                # Wrap-around (re-shuffle)
                random.shuffle(dom_idx[d])
                state["dom_ptrs"][d] = 0
                ptr = 0
            sampled.append(dom_idx[d][ptr])
            state["dom_ptrs"][d] += 1
            attempts += 1
        return sampled

    def _build_batches(self):
        all_idx = []
        for _ in range(self.iters):
            # Sample P classes
            if self.n_classes >= self.P:
                cls_batch = random.sample(self.all_classes, self.P)
            else:
                cls_batch = self.all_classes + random.choices(
                    self.all_classes, k=self.P - self.n_classes
                )
            for c in cls_batch:
                idxs = self._sample_class(c)
                if len(idxs) < self.K:
                    # Pad by repeat (rare — only if class has very few samples)
                    while len(idxs) < self.K:
                        idxs.append(idxs[len(idxs) % max(1, len(idxs))])
                all_idx.extend(idxs)
        return np.array(all_idx)

    def __iter__(self):
        return iter(self._build_batches())

    def __len__(self):
        return self.iters * self.bs
