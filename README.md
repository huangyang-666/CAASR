# CAASR: A Real-World Animation Super-Resolution Benchmark with Color Degradation and Multi-Scale Multi-Frequency Alignment (TIP 2025)
CAASR:  
CAASR is a benchmark designed to advance the frontier of animation super-resolution. It features a high-quality dataset and a dedicated training pipeline for both 2D and 3D animated content, emphasizing color degradation, frequency alignment, and scale adaptability.

:dart: Update：

- :ok_hand: Tools Will Coming Soon
- :white_check_mark: 2025.7.26  Pretrained Weight Added
- :white_check_mark: 2025.7.25  Data Added
- :white_check_mark: 2025.7.21  Code Added
  
:book: Visualization：

:surfer: Installation：

:innocent: ADASR Dataset：  
 - The complete training dataset, encompassing both 2D and 3D animation sequences, is available via  
[Baidu Drive](https://pan.baidu.com/s/1wLWdVZdZhgL2OO2ADaWlLw) (code: `a135`)

:hearts: Fast Inference：  
 - The evaluation dataset can be accessed through  
[Baidu Drive](https://pan.baidu.com/s/1eJf7BE3VUb-3LebW_M5weQ) (code: `a135`)

:clubs: Pretrain Weight：  
 - Pretrained models for both 2D and 3D animation are available via  
[Baidu Drive](https://pan.baidu.com/s/15eUi6gR8jhOxj3Q9OQDpYQ) (code: `a135`) | [Google Drive](https://drive.google.com/drive/folders/1m8hNSFWLprjF1EO4jIwE9yEEOLKch9NW?usp=drive_link).
 - Weights for comparison methods can be found here:  
[Baidu Drive](https://pan.baidu.com/s/1vOxc1WJBe0TjKwrwexp9EQ) (code: `a135`) | [Google Drive](https://drive.google.com/drive/folders/1MMld17E4Q6DcbUSENTQdiprEVKBp5t1Y?usp=drive_link)

:tophat: Train：  
 - We recommend configuring degradation strategies and selecting scale-aware architectures based on the characteristics of your content and downstream tasks.

:beer: Test：  
 - We adopt [PYIQA](https://github.com/chaofengc/IQA-PyTorch) for perceptual quality assessment. For 2D animation, default configurations are applied. For 3D content, we provide fine-tuned versions of MANIQA and TReS. The corresponding model weights are available here:  
[Baidu Drive](https://pan.baidu.com/s/1r071-EV3nqnrdmkoXgI8DQ) (code: `a135`) | [Google Drive](https://drive.google.com/drive/folders/115nK64ch1jTkqC3YqhDw9Z6eSAiVO6KK?usp=drive_link)

:wrench: Tools：

:chocolate_bar: Citation：

```bibtex
@article{jiang2025animationSR,
  title   = {{A Real-World Animation Super-Resolution Benchmark with Color Degradation and Multi-Scale Multi-Frequency Alignment}},
  author  = {Jiang, Yu and Zhang, Yongji and Li, Siqi and Huang, Yang and Wang, Yuehang and Yao, Yutong and Gao, Yue},
  journal = {IEEE Transactions on Image Processing},
  year    = {2025}
}


:house: License：

:rocket: Acknowledgement：

:airplane: This project remains under continuous development. 
 - We invite the community to explore and expand the use of state-of-the-art super-resolution techniques in both academic and production-grade animation workflows.
