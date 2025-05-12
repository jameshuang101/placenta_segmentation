# Motivation & Objective
* **Clinical need**: Accurate 3D segmentation of the placenta and uterine cavity on prenatal MRI can aid in diagnosing placenta accreta spectrum (PAS) and other complications, overcoming the limitations of 2D ultrasound .

* **Goal**: Develop a fully-automatic, rapid, and accurate 3D deep-learning segmentation pipeline for both sagittal and axial T2-weighted MR volumes.

# Data & Preprocessing
* **Datasets**:

  * Sagittal: 101 normal pregnant women (27–80 slices, 256×256 px).

  * Axial: 143 PAS suspects + 101 normals (28–62 slices, 256×256 px).

* **Splits**: 70% train, 10% val, 20% test for each view.

* **Cropping**: Zero-pad small volumes to 256×256, then crop along slice axis to the region containing the uterus (24–61 slices) based on manual labels.

* **Denoising & normalization**: 2D median filter (3×3), then intensity clipped to [p5, p99] and linearly scaled to [0,1].

* **Label encoding**: Three channels—background, placenta, uterus-without-placenta—via one-hot stacking.

* **Blocking**: Generate overlapping 3D blocks of five consecutive slices (block stride = 1) for training (N–4 blocks per volume).

# Network Architecture
* **3D U-Net 3+ (modified)**:

  * Input: 256×256×5×1 blocks

  * Output: 256×256×5×3 (softmax over background, placenta, uterus)

* **Changes vs. original UNet 3+**:

  * Reduced from five to four resolution levels for memory efficiency.

  * Full-scale skip connections integrate low- and high-level features for improved boundary and positional awareness.

* **Loss**: Hybrid Dice loss

# Training & Post-processing
* **Hardware**: 4× NVIDIA RTX A6000 GPUs; batch size = 32 (8 blocks/GPU via MirroredStrategy).

* **Optimizer**: Adam, lr = 1e-4; up to 500 epochs with best-val-loss checkpointing.

* **Inference**: Overlap-tiling of 5-slice blocks with 4-slice overlap; horizontal flip augmentation averaged with original.

* **Post-processing (in MATLAB)**:

  1. 3D erosion (3×3×3) to remove protrusions

  2. Keep largest 26-connected component

  3. 3D dilation (3×3×3)

  4. Fill holes
 
# Results
### Segmentation Accuracy
| Dataset  | Structure      | U-Net 3+ DSC (%) ±SD | Previous U-Net DSC (%) ±SD |
| -------- | -------------- | -------------------- | -------------------------- |
| Sagittal | Placenta       | **87.7 ± 3.1**       | 83.4 ± 5.7                 |
|          | Uterine cavity | **95.3 ± 1.4**       | 88.6 ± 3.4                 |
| Axial    | Placenta       | **82.7 ± 5.1**       | 81.0 ± 7.2                 |
|          | Uterine cavity | **91.8 ± 3.3**       | 87.2 ± 5.6                 |

### Robustness & Speed
* **Inference time**: ≈ 30 s per full volume.

* **Error trends**:

  * Under-segmentation of placenta in axial cases, especially post-hysterectomy, due to anatomical variability.

  * Over-segmentation tendency in uterine cavity under PAS conditions.
 
# Conclusions & Future Directions
* **Efficacy**: The modified 3D U-Net 3+ yields high‐accuracy, fast, fully‐automatic 3D segmentation of placenta and uterine cavity, outperforming prior 3D U-Net models.

* **Limitations**: Memory constraints limit block depth (5 slices); anatomical variability in PAS remains challenging.

* **Next steps**:

  * Incorporate larger 3D context (wider blocks) for improved performance.

  * Explore advanced post‐processing (graph-cuts, CRFs) and additional augmentations.

  * Quantitatively compare sagittal vs. axial MRI for specific clinical tasks.

# Acknowledgements
This work is directly related to the paper *J. Huang, M. Shahedi, Q. Do, Y. Xi, M. Lewis, C. Herrera, D. Owen, C. Spong, A. Madhuranthakam, D. Twickler, and B. Fei, “Deep-learning based segmentation of the placenta and uterine cavity on prenatal MR images.” In Medical Imaging 2023: Computer-Aided Diagnosis, vol. 12465, pp. 12465ON, SPIE, 2023.*

**Paper**: [SPIE_2023_PlacentaSegmentation_SPIE.pdf](https://github.com/JamesHuang404/DL-Placenta-Segmentation/files/11186342/SPIE_2023_PlacentaSegmentation_SPIE.pdf)

**Presentation Video**: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12465/124650N/Deep-learning-based-segmentation-of-the-placenta-and-uterine-cavity/10.1117/12.2653659.full

