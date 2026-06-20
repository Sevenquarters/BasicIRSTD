# DNANet 系列一页版答辩摘要

## 总图入口

- PNG 版：[dnanet_family_onepage_overview.png](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\figures\dnanet_family_summary\dnanet_family_onepage_overview.png)
- PDF 版：[dnanet_family_onepage_overview.pdf](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\figures\dnanet_family_summary\dnanet_family_onepage_overview.pdf)

---

## 一句话结论

- `LDEM` 证明了浅层细节增强有效，`LDEM-Gate` 进一步证明了选择性筛选能提升峰值性能，但 `IRSTD-1K` 上仍存在后期稳定性不足的问题。

---

## 答辩时最值得讲的三点

### 1. 方法递进是成立的

- `DNANet` 是强 baseline
- `DNANet-LDEM` 说明浅层细节增强是有效的
- `DNANet-LDEM-Gate` 说明“增强后再筛选”能继续提升峰值表现

### 2. 两个数据集给出的科研信号不同

- `NUDT-SIRST` 上，第二版已经兼顾性能和稳定性
- `IRSTD-1K` 上，第二版虽然把 best mIoU 提高到了 `0.6522`，但 final mIoU 回落到 `0.5232`
- 这说明当前最重要的问题不再是“如何继续提峰值”，而是“如何稳住后期表现”

### 3. 第三版方向已经明确

- 第三版不建议继续盲目堆更强的增强模块
- 更适合做 `DNANet-LDEM-Gate-Stable`
- 重点是：门控稳定化、语义反校正、跨层一致性约束

---

## 最简结果口径

### NUDT-SIRST

| 模型 | Best mIoU | Final mIoU |
|---|---:|---:|
| DNANet | 0.8118 | 0.8114 |
| DNANet-LDEM | 0.8409 | 0.8218 |
| DNANet-LDEM-Gate | 0.8397 | 0.8395 |

### IRSTD-1K

| 模型 | Best mIoU | Final mIoU |
|---|---:|---:|
| DNANet | 0.6446 | 0.6282 |
| DNANet-LDEM | 0.6499 | 0.6373 |
| DNANet-LDEM-Gate | 0.6522 | 0.5232 |

---

## 如果只展示一张图

- 直接展示：[dnanet_family_onepage_overview.png](D:\Program Files (x86)\IRSTD\BasicIRSTD\docs\experiments\figures\dnanet_family_summary\dnanet_family_onepage_overview.png)
- 这张图已经包含：
  - 两个数据集的总表
  - best/final 对比柱状图
  - 稳定性 gap 对比
  - 结论摘要
