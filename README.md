# better-synth-dxm

## 1 赛题流程

step 1. 依赖安装

```shell
conda create -n dj python=3.10
conda activate dj

bash install.sh
```

step 2. 比赛资源下载

- 下载基础模型，种子/微调/评测数据集
- 基础模型与微调数据集均存放于训练目录中指定位置
- 种子数据集存放于`input`目录
- 评测数据集存放于`toolkit/eval`目录

```shell
bash download.sh
```

step 3. 运行
准备好对应的数据集, 修改 'toolkit/train_mgm_2b_stage_1.sh'中对应的数据集路径, 进行训练.

```shell
bash train_mgm_2b_stage_1.sh
```

训练模型权重和日志文件会保存在output目录下.

## 2 方案介绍

### 2.1 赛题介绍

Better Synth 是一项以数据为中心的挑战赛，考察如何合成与清洗图文数据以在多模态大模型上取得更优的图片理解能力。初赛给 400k
数据, 400k 选 200k 数据作为预训练阶段数据参与训练.
复赛提供 10k 数据, 10k 选 5k 数据作为预训练阶段数据, 在初赛产出模型基础上重新训练.

### 2.2 方案介绍

下面是我们答辩提交方案, 有的地方(如采用 diffusion 方案进行图像生成)是我们认为会有效果但是实际没有做的.
在复赛开始, 我们认识到数据多样性的重要性, 因此我们的方案侧重于保留初始数据的多样性, 整体流程如下图所示:

1. 根据 caption 进行聚类, 得到多个簇, 即每条 caption, 都会对应一个簇. 我们希望在构造数据时, 从这些簇中选取数据,
   且尽可能保持原始簇的数量比例;
2. 使用 data-juicer 中的算子进行数据过滤, 将数据分为了 X_low 和 X_high 两部分;
3. 使用 InternVL-2-40B 全部做简短的 caption 重写;
4. 对于 X_high 部分的数据, 我们默认它已经是高质量数据, 可以结合 IT 方案(详见论文 or Github 链接) 做进一步的扩写+幻觉消除.
   而对于 X_low 这部分的数据, 我们直接根据图片 id, 在 重写 caption 中查找并替换;
5. 根据原始簇比例, 对池中数据再次按比例进行筛选. 优先保留 X_high 中的数据.最终得到满足要求的 5k 条训练数据进行训练.

FAQ:

- 为什么要用简短的 caption 重写?
  实验中发现, 尽可能详细且带有逻辑关系的 caption 在实际训练中效果并不好, 甚至不如原始 caption. 我们推测:

1) 模型本身参数量过小, 学习过多信息对它会有比较大的负担;
2) 训练参数 epoch=1, 导致模型训练不充分.
   基于以上考虑, 仅针对本次比赛而言, 通过 prompt 约束模型在合成 caption 时描述应尽量简洁, 突出主要内容,
   只存在描述性话语且避免让模型推测.

- 哪些图片最需要使用 Diffusion 进行重写?
  我们在观察中发现了一些 badcase, 如水印内容, 色情图片等, 这些图片有必要被过滤出来并重新根据 caption 进行重写,
  以获取更好的图片-文字对.

- data-juicer 数据筛选部分
  在初赛时, 我们直接使用了 data-juicer 给出的 llava recipe, 针对赛题数据进行分析后, 将关键指标: image-text similarity 和
  image-text matching 的阈值设置为 25 分位点. 其他指标根据 3 sigma 原则过滤异常值即可.

### 2.3 其他方案

#### 2.3.1 VILA^2

论文链接: https://arxiv.org/html/2407.17453v1

我们尝试复现VILA^2的方案, 但是根据文章给的prompt, 会**产生比较长的caption**.

此外, 没有找到能针对本次赛题的模型MGM-2B进行加速的框架, VLLM, LMDeploy等都无法满足需求(最终直接使用的model_vqa.py修改),
由于缺乏加速框架,
我们认为使用它在数据合成时迭代多轮用时会过长.

但是, 我们仍认为它是一种有效的方案: 按照我们的理解, 相比于强模型提供的caption, 训练时使用自己产出的caption数据会更容易学习.

### 2.3.2 Image-Textualization
论文链接: https://arxiv.org/html/2303.17207v1
即上文介绍的it方案, 我们在初赛尝试使用它作为数据合成的步骤之一, 作者给出的卖点主要在于幻觉消除.
原始数据本身就比较简短, 在合成之后使用效果会更好. 使用步骤详见: /solution/image_textualization/README.md


## 3 运行

#### step1 准备好数据

我们建议在这一步先做一个基本的过滤, 比如只过滤掉异常图片或者caption相关性过低的图片.
便于更精准的聚类. 直接聚类也可以.

```
cd toolkit/data-juicer
python tools/process_data.py --config ./configs/xx.yaml
```

#### step2 将数据聚类为簇

将脚本参数修改为自己的数据集路径

```shell
bash solution/cluster/cluster.sh
```

#### step3 数据合成

经过step2后, 得到数据格式如下所示:

```

```

由于复赛数据量较小, 我们进行了全量合成, 然后再筛选出2.5k数据double一下作为训练数据.
可以根据实际需要 先进行step4, 然后把需要合成的数据筛选出来进行合成.
注意修改里面的相关文件路径

```
python solution/data_syn/batch_infer_intern_vl.py
```

#### step4 数据筛选


#### step5 进行训练
修改对应路径名,进行训练
```
bash toolkit/train_mgm_2b_stage_1.sh
```
