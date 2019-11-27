import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):
    _ops: nn.ModuleList

    def __init__(self, C: int, stride: int):
        super().__init__()
        self._ops = nn.ModuleList()

        for primitive in PRIMITIVES:
            op: nn.Module = OPS[primitive](C, stride, False)

            if 'pool' in primitive:  # avg_pool_3x3 or max_pool_3x3
                # poolingがあったらとりあえず思考停止でBatchNorm2dかませる。
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        # 雰囲気的には、各operationの選択確率がweightsとして渡されるから、
        # それをcontinuous domain(連続値の空間)に落とし込んでいるようにみえる。
        # TODO(c-bata): nn.Module と weight の積が何を意味するか確認
        s = sum(w * op(x) for w, op in zip(weights, self._ops))
        # p weights[0]: [torch.cuda.FloatTensor of size 1 (GPU 0)]
        """ weights は最初こんな感じ
        (Pdb) p weights
        Variable containing:
         0.1249
         0.1252
         0.1250
         0.1249
         0.1250
         0.1250
         0.1250
         0.1251
        [torch.cuda.FloatTensor of size 8 (GPU 0)]
        (pdb) p sum(weights)
        Variable containing:
         1
        [torch.cuda.FloatTensor of size 1 (GPU 0)]
        """

        # x: torch.cuda.FloatTensor of size 64x16x32x32 (GPU 0)]
        # self._ops[0](x): [torch.cuda.FloatTensor of size 64x16x32x32 (GPU 0)]
        # s: [torch.cuda.FloatTensor of size 64x16x32x32 (GPU 0)]
        # weights は スカラーだから ops[0](x) の次元を変えない
        return s


class Cell(nn.Module):
    preprocess0: nn.Module
    preprocess1: nn.Module
    _steps: int
    _multiplier: int
    _ops: nn.ModuleList  # MixedOp がappendされていく。
    _bns: nn.ModuleList

    def __init__(
            self,
            steps: int,
            multiplier: int,  # 今のところ用途が謎。Networkクラスで使ってるっぽい。
            C_prev_prev: int,  # 2つ前のcellの出力の次元っぽい
            C_prev: int,  # 1つ前のcellの出力の次元っぽい
            C: int,  # 自身のCellの入力 or 出力の次元。どっちかはFactorizedReduceが何やってるか調べたらいいかも。
            reduction: bool,  # strideを2にして次元を落とすかを指定
            reduction_prev: bool,  # 1つ前のcellがstrideを2にして次元を落としたかどうかを保持
    ):
        super().__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier :], dim=1)


class Network(nn.Module):
    def __init__(
        self,
        C,
        num_classes,
        layers,
        criterion,  # criterion=基準. nn.CrossEntropyLoss() とかが渡されてくる.
        steps=4,
        multiplier=4,
        stem_multiplier=3,  # Conv2d層で何倍に次元を増やすかを指定。多いほどおそらく表現力が上がるけど、学習が難しくなるんだと思う
    ):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        # 入力 3次元、出力 C_curr 次元の畳み込み層は必ず最初に通すようにしている。つまりこの層はアーキテクチャの探索を行わない。
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # if i in [2, 5]: 必ず2つreductionレイヤーを入れるようにしてる。
                """
                >>> def print_reduction(layers):
                ...     for i in range(layers):
                ...         hits = [layers // 3, 2 * layers // 3]
                ...         print(f"layer_{i}: {i in hits} : {hits}")
                ...
                >>> print_reduction(8)
                layer_0: False : [2, 5]
                layer_1: False : [2, 5]
                layer_2: True : [2, 5]
                layer_3: False : [2, 5]
                layer_4: False : [2, 5]
                layer_5: True : [2, 5]
                layer_6: False : [2, 5]
                layer_7: False : [2, 5]
                >>> print_reduction(12)
                layer_0: False : [4, 8]
                layer_1: False : [4, 8]
                layer_2: False : [4, 8]
                layer_3: False : [4, 8]
                layer_4: True : [4, 8]
                layer_5: False : [4, 8]
                layer_6: False : [4, 8]
                layer_7: False : [4, 8]
                layer_8: True : [4, 8]
                layer_9: False : [4, 8]
                layer_10: False : [4, 8]
                layer_11: False : [4, 8]
                """
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                steps,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
            )
            reduction_prev = reduction
            self.cells += [cell]
            # なぜか C_prev は C_curr に multiplier をかけている。
            # 1つ前のCellの次元は、今回の入力次元にmultiplier(4)をかけたものらしい。

            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(
            self._C, self._num_classes, self._layers, self._criterion
        ).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        # CIFAR 10 の画像データセットは、 32x32 の画像が60000枚入っていて、それぞれ10クラスに分類されている。
        # RGBの3 channelがあるっぽい。
        # input: [torch.cuda.FloatTensor of size 64x3x32x32 (GPU 0)]

        s0 = s1 = self.stem(input)
        # s0: [torch.cuda.FloatTensor of size 64x48x32x32 (GPU 0)]

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                # self.alphas_normal: [torch.cuda.FloatTensor of size 14x8 (GPU 0)]
                weights = F.softmax(self.alphas_normal, dim=-1)
            # weights: [torch.cuda.FloatTensor of size 14x8 (GPU 0)]

            s0, s1 = s1, cell(s0, s1, weights)
            # s0: [torch.cuda.FloatTensor of size 64x48x32x32 (GPU 0)]
            # s1: [torch.cuda.FloatTensor of size 64x64x32x32 (GPU 0)]
        out = self.global_pooling(s1)
        # out: [torch.cuda.FloatTensor of size 64x256x1x1 (GPU 0)]
        # out.view(out.size(0), -1): [torch.cuda.FloatTensor of size 64x256 (GPU 0)]

        logits = self.classifier(out.view(out.size(0), -1))
        # logits: [torch.cuda.FloatTensor of size 64x10 (GPU 0)]

        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(
            1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True
        )
        self.alphas_reduce = Variable(
            1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True
        )
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(
                        W[x][k]
                        for k in range(len(W[x]))
                        if k != PRIMITIVES.index("none")
                    ),
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index("none"):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype
