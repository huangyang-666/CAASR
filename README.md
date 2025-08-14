# CAASR: A Real-World Animation Super-Resolution Benchmark with Color Degradation and Multi-Scale Multi-Frequency Alignment (TIP 2025)

**CAASR** is a benchmark designed to advance the frontier of animation super-resolution.  
It features a high-quality dataset and a dedicated training pipeline for both 2D and 3D animated content, with a focus on **color degradation**, **frequency alignment**, and **scale adaptability**.

---

## :dart: Updates
- :ok_hand: Tools Coming Soon
- :white_check_mark: **2025.07.26** – Pretrained Weights Added
- :white_check_mark: **2025.07.25** – Dataset Added
- :white_check_mark: **2025.07.21** – Code Released

---

## :book: Visualization
*(Coming Soon)*

---

## :surfer: Installation
*(Instructions Coming Soon)*

---

## :innocent: ADASR Dataset
- **Full Training Dataset** (2D & 3D animation sequences):  
  [Baidu Drive](https://pan.baidu.com/s/1wLWdVZdZhgL2OO2ADaWlLw) (code: `a135`)

---

## :hearts: Fast Inference
- **Evaluation Dataset**:  
  [Baidu Drive](https://pan.baidu.com/s/1eJf7BE3VUb-3LebW_M5weQ) (code: `a135`)

---

## :clubs: Pretrained Weights
- **Pretrained Models** (2D & 3D animation):  
  [Baidu Drive](https://pan.baidu.com/s/15eUi6gR8jhOxj3Q9OQDpYQ) (code: `a135`) | [Google Drive](https://drive.google.com/drive/folders/1m8hNSFWLprjF1EO4jIwE9yEEOLKch9NW?usp=drive_link)  
- **Weights for Comparison Methods**:  
  [Baidu Drive](https://pan.baidu.com/s/1vOxc1WJBe0TjKwrwexp9EQ) (code: `a135`) | [Google Drive](https://drive.google.com/drive/folders/1MMld17E4Q6DcbUSENTQdiprEVKBp5t1Y?usp=drive_link)

---

## :tophat: Training
- Configure degradation strategies and choose scale-aware architectures according to your animation content and downstream tasks.

---

## :beer: Testing
- We adopt [PYIQA](https://github.com/chaofengc/IQA-PyTorch) for perceptual quality assessment.
- **2D Animation**: Default configurations are applied.  
- **3D Animation**: Fine-tuned MANIQA and TReS models are provided.  
  [Baidu Drive](https://pan.baidu.com/s/1r071-EV3nqnrdmkoXgI8DQ) (code: `a135`) | [Google Drive](https://drive.google.com/drive/folders/115nK64ch1jTkqC3YqhDw9Z6eSAiVO6KK?usp=drive_link)

---

## :wrench: Tools
*(Coming Soon)*

---

## :chocolate_bar: Citation

If you find our work useful, please consider citing:

```bibtex
@article{animationSR,
  title   = {{A Real-World Animation Super-Resolution Benchmark with Color Degradation and Multi-Scale Multi-Frequency Alignment}},
  author  = {Jiang, Yu and Zhang, Yongji and Li, Siqi and Huang, Yang and Wang, Yuehang and Yao, Yutong and Gao, Yue},
  journal = {IEEE Transactions on Image Processing},
  year    = {2025}
}
