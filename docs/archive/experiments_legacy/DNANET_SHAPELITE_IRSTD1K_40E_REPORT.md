# DNANet-LDEM-Gate-Stable-ShapeLite 在 IRSTD-1K 上的 40 Epoch 结果

## 1. 实验配置

- 模型：`DNANet-LDEM-Gate-Stable-ShapeLite`
- 数据集：`IRSTD-1K`
- 训练方式：`10 -> 20 -> 30 -> 40` 分段续跑
- 参数：
  - `batchSize = 8`
  - `patchSize = 256`
  - `eval_intervals = 1`
  - `save_intervals = 10`

结果文件：

- [metrics.csv](D:\Program Files (x86)\IRSTD\BasicIRSTD\log\exp_dnanet_ldem_gate_stable_shapelite_irstd1k_40e\IRSTD-1K_DNANet-LDEM-Gate-Stable-ShapeLite_metrics.csv)
- [best checkpoint](D:\Program Files (x86)\IRSTD\BasicIRSTD\log\exp_dnanet_ldem_gate_stable_shapelite_irstd1k_40e\IRSTD-1K\DNANet-LDEM-Gate-Stable-ShapeLite_best.pth.tar)
- [epoch 40 checkpoint](D:\Program Files (x86)\IRSTD\BasicIRSTD\log\exp_dnanet_ldem_gate_stable_shapelite_irstd1k_40e\IRSTD-1K\DNANet-LDEM-Gate-Stable-ShapeLite_40.pth.tar)


## 2. ShapeLite 结果

### Best

- Best Epoch：`29`
- Best mIoU：`0.6470`
- Best PD：`0.9057`
- Best FA：`3.5034e-05`

### Final

- Final Epoch：`40`
- Final mIoU：`0.5870`
- Final PD：`0.9158`
- Final FA：`7.1227e-05`

### Stability Gap

- `0.6470 - 0.5870 = 0.0600`


## 3. 与已有版本对比

| 模型 | Best mIoU | Final mIoU | Stability Gap | 判断 |
|---|---:|---:|---:|---|
| DNANet | 0.6446 | 0.6282 | 0.0163 | baseline 稳定 |
| DNANet-LDEM | 0.6499 | 0.6373 | 0.0125 | 最均衡之一 |
| DNANet-LDEM-Gate | 0.6522 | 0.5232 | 0.1290 | 峰值高但崩塌严重 |
| DNANet-LDEM-Gate-Stable | 0.6387 | 0.6331 | 0.0055 | 最稳定 |
| DNANet-LDEM-Gate-Stable-Shape | 0.6420 | 0.6192 | 0.0227 | 比 Gate 稳，但不如 Stable |
| DNANet-LDEM-Gate-Stable-ShapeLite | 0.6470 | 0.5870 | 0.0600 | 峰值提升，但后期仍不稳 |


## 4. 结论

- `ShapeLite` 比强 `Shape` 更温和，前中期训练更顺。
- `ShapeLite` 的 best mIoU 达到 `0.6470`，高于 `Stable` 的 `0.6387`。
- 但 `ShapeLite` 的 final mIoU 只有 `0.5870`，明显低于 `Stable` 的 `0.6331`。
- 说明当前 `ShapeLite` 虽然弱化了 shape 干预，但仍然会在后期破坏稳定保持能力。

一句话判断：

- `Stable` 仍然是当前最适合作为主模型的版本；
- `ShapeLite` 证明“弱 shape 先验”有峰值收益，但还没有把这种收益稳定保留下来。
