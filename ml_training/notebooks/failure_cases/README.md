# Failure Cases (KADID-10k Test Split)

For each issue head, the 8 test images where the model's predicted probability was furthest from the true binary label (confident misses and confident false alarms both surface here). For quality_score, the 8 images with the largest absolute prediction error.

## blur

- ![blur_I76_03_03.png](blur_I76_03_03.png) `I76_03_03.png` (dist: Motion blur lvl 3) -- true_label=1, predicted_prob=0.016
- ![blur_I72_03_03.png](blur_I72_03_03.png) `I72_03_03.png` (dist: Motion blur lvl 3) -- true_label=1, predicted_prob=0.029
- ![blur_I75_03_03.png](blur_I75_03_03.png) `I75_03_03.png` (dist: Motion blur lvl 3) -- true_label=1, predicted_prob=0.032
- ![blur_I76_01_03.png](blur_I76_01_03.png) `I76_01_03.png` (dist: Gaussian blur lvl 3) -- true_label=1, predicted_prob=0.041
- ![blur_I76_03_05.png](blur_I76_03_05.png) `I76_03_05.png` (dist: Motion blur lvl 5) -- true_label=1, predicted_prob=0.042
- ![blur_I75_03_04.png](blur_I75_03_04.png) `I75_03_04.png` (dist: Motion blur lvl 4) -- true_label=1, predicted_prob=0.055
- ![blur_I21_02_02.png](blur_I21_02_02.png) `I21_02_02.png` (dist: Lens blur lvl 2) -- true_label=0, predicted_prob=0.935
- ![blur_I75_01_03.png](blur_I75_01_03.png) `I75_01_03.png` (dist: Gaussian blur lvl 3) -- true_label=1, predicted_prob=0.088

## underexposure

- ![underexposure_I52_17_03.png](underexposure_I52_17_03.png) `I52_17_03.png` (dist: Darken lvl 3) -- true_label=1, predicted_prob=0.000
- ![underexposure_I52_17_04.png](underexposure_I52_17_04.png) `I52_17_04.png` (dist: Darken lvl 4) -- true_label=1, predicted_prob=0.001
- ![underexposure_I24_17_04.png](underexposure_I24_17_04.png) `I24_17_04.png` (dist: Darken lvl 4) -- true_label=1, predicted_prob=0.003
- ![underexposure_I24_17_03.png](underexposure_I24_17_03.png) `I24_17_03.png` (dist: Darken lvl 3) -- true_label=1, predicted_prob=0.003
- ![underexposure_I76_17_03.png](underexposure_I76_17_03.png) `I76_17_03.png` (dist: Darken lvl 3) -- true_label=1, predicted_prob=0.005
- ![underexposure_I75_17_05.png](underexposure_I75_17_05.png) `I75_17_05.png` (dist: Darken lvl 5) -- true_label=1, predicted_prob=0.009
- ![underexposure_I76_17_04.png](underexposure_I76_17_04.png) `I76_17_04.png` (dist: Darken lvl 4) -- true_label=1, predicted_prob=0.011
- ![underexposure_I72_17_03.png](underexposure_I72_17_03.png) `I72_17_03.png` (dist: Darken lvl 3) -- true_label=1, predicted_prob=0.021

## overexposure

- ![overexposure_I52_16_03.png](overexposure_I52_16_03.png) `I52_16_03.png` (dist: Brighten lvl 3) -- true_label=1, predicted_prob=0.000
- ![overexposure_I52_16_04.png](overexposure_I52_16_04.png) `I52_16_04.png` (dist: Brighten lvl 4) -- true_label=1, predicted_prob=0.001
- ![overexposure_I21_16_03.png](overexposure_I21_16_03.png) `I21_16_03.png` (dist: Brighten lvl 3) -- true_label=1, predicted_prob=0.016
- ![overexposure_I61_16_03.png](overexposure_I61_16_03.png) `I61_16_03.png` (dist: Brighten lvl 3) -- true_label=1, predicted_prob=0.019
- ![overexposure_I24_16_03.png](overexposure_I24_16_03.png) `I24_16_03.png` (dist: Brighten lvl 3) -- true_label=1, predicted_prob=0.021
- ![overexposure_I75_16_03.png](overexposure_I75_16_03.png) `I75_16_03.png` (dist: Brighten lvl 3) -- true_label=1, predicted_prob=0.045
- ![overexposure_I76_16_03.png](overexposure_I76_16_03.png) `I76_16_03.png` (dist: Brighten lvl 3) -- true_label=1, predicted_prob=0.049
- ![overexposure_I15_16_03.png](overexposure_I15_16_03.png) `I15_16_03.png` (dist: Brighten lvl 3) -- true_label=1, predicted_prob=0.069

## noise

- ![noise_I52_15_03.png](noise_I52_15_03.png) `I52_15_03.png` (dist: Denoise lvl 3) -- true_label=1, predicted_prob=0.001
- ![noise_I52_14_03.png](noise_I52_14_03.png) `I52_14_03.png` (dist: Multiplicative noise lvl 3) -- true_label=1, predicted_prob=0.001
- ![noise_I52_13_03.png](noise_I52_13_03.png) `I52_13_03.png` (dist: Impulse noise lvl 3) -- true_label=1, predicted_prob=0.001
- ![noise_I52_11_03.png](noise_I52_11_03.png) `I52_11_03.png` (dist: White noise lvl 3) -- true_label=1, predicted_prob=0.002
- ![noise_I52_12_03.png](noise_I52_12_03.png) `I52_12_03.png` (dist: White noise in color component lvl 3) -- true_label=1, predicted_prob=0.005
- ![noise_I52_11_04.png](noise_I52_11_04.png) `I52_11_04.png` (dist: White noise lvl 4) -- true_label=1, predicted_prob=0.006
- ![noise_I52_14_04.png](noise_I52_14_04.png) `I52_14_04.png` (dist: Multiplicative noise lvl 4) -- true_label=1, predicted_prob=0.006
- ![noise_I52_14_05.png](noise_I52_14_05.png) `I52_14_05.png` (dist: Multiplicative noise lvl 5) -- true_label=1, predicted_prob=0.012

## corruption

- ![corruption_I21_20_05.png](corruption_I21_20_05.png) `I21_20_05.png` (dist: Non-eccentricity patch lvl 5) -- true_label=1, predicted_prob=0.040
- ![corruption_I21_18_05.png](corruption_I21_18_05.png) `I21_18_05.png` (dist: Mean shift lvl 5) -- true_label=1, predicted_prob=0.056
- ![corruption_I52_22_03.png](corruption_I52_22_03.png) `I52_22_03.png` (dist: Quantization lvl 3) -- true_label=0, predicted_prob=0.943
- ![corruption_I52_22_01.png](corruption_I52_22_01.png) `I52_22_01.png` (dist: Quantization lvl 1) -- true_label=0, predicted_prob=0.932
- ![corruption_I52_22_04.png](corruption_I52_22_04.png) `I52_22_04.png` (dist: Quantization lvl 4) -- true_label=0, predicted_prob=0.930
- ![corruption_I21_06_03.png](corruption_I21_06_03.png) `I21_06_03.png` (dist: Color quantization lvl 3) -- true_label=1, predicted_prob=0.072
- ![corruption_I52_09_02.png](corruption_I52_09_02.png) `I52_09_02.png` (dist: JPEG2000 lvl 2) -- true_label=0, predicted_prob=0.927
- ![corruption_I52_08_03.png](corruption_I52_08_03.png) `I52_08_03.png` (dist: Color saturation 2 lvl 3) -- true_label=0, predicted_prob=0.925

## quality_score

- ![quality_score_I52.png](quality_score_I52.png) `I52.png` (dist: none lvl 0) -- true_quality=100.0, predicted_quality=4.2
- ![quality_score_I52_18_02.png](quality_score_I52_18_02.png) `I52_18_02.png` (dist: Mean shift lvl 2) -- true_quality=95.8, predicted_quality=3.6
- ![quality_score_I52_01_01.png](quality_score_I52_01_01.png) `I52_01_01.png` (dist: Gaussian blur lvl 1) -- true_quality=94.2, predicted_quality=4.2
- ![quality_score_I52_18_04.png](quality_score_I52_18_04.png) `I52_18_04.png` (dist: Mean shift lvl 4) -- true_quality=93.2, predicted_quality=4.9
- ![quality_score_I52_03_01.png](quality_score_I52_03_01.png) `I52_03_01.png` (dist: Motion blur lvl 1) -- true_quality=92.5, predicted_quality=4.2
- ![quality_score_I52_08_01.png](quality_score_I52_08_01.png) `I52_08_01.png` (dist: Color saturation 2 lvl 1) -- true_quality=92.5, predicted_quality=4.2
- ![quality_score_I52_16_01.png](quality_score_I52_16_01.png) `I52_16_01.png` (dist: Brighten lvl 1) -- true_quality=91.8, predicted_quality=3.8
- ![quality_score_I52_18_01.png](quality_score_I52_18_01.png) `I52_18_01.png` (dist: Mean shift lvl 1) -- true_quality=90.8, predicted_quality=3.1
