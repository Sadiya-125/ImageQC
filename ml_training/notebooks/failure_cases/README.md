# Failure Cases (KADID-10k Test Split)

For each issue head, the 8 test images where the model's predicted probability was furthest from the true binary label (confident misses and confident false alarms both surface here). For quality_score, the 8 images with the largest absolute prediction error.

## blur

- ![blur_I72_03_03.png](blur_I72_03_03.png) `I72_03_03.png` (dist: Motion blur lvl 3) -- true_label=1, predicted_prob=0.024
- ![blur_I15_03_03.png](blur_I15_03_03.png) `I15_03_03.png` (dist: Motion blur lvl 3) -- true_label=1, predicted_prob=0.032
- ![blur_I15_01_03.png](blur_I15_01_03.png) `I15_01_03.png` (dist: Gaussian blur lvl 3) -- true_label=1, predicted_prob=0.036
- ![blur_I61_03_03.png](blur_I61_03_03.png) `I61_03_03.png` (dist: Motion blur lvl 3) -- true_label=1, predicted_prob=0.043
- ![blur_I72_01_03.png](blur_I72_01_03.png) `I72_01_03.png` (dist: Gaussian blur lvl 3) -- true_label=1, predicted_prob=0.060
- ![blur_I61_03_04.png](blur_I61_03_04.png) `I61_03_04.png` (dist: Motion blur lvl 4) -- true_label=1, predicted_prob=0.077
- ![blur_I76_03_03.png](blur_I76_03_03.png) `I76_03_03.png` (dist: Motion blur lvl 3) -- true_label=1, predicted_prob=0.079
- ![blur_I21_03_03.png](blur_I21_03_03.png) `I21_03_03.png` (dist: Motion blur lvl 3) -- true_label=1, predicted_prob=0.081

## underexposure

- ![underexposure_I52_17_03.png](underexposure_I52_17_03.png) `I52_17_03.png` (dist: Darken lvl 3) -- true_label=1, predicted_prob=0.002
- ![underexposure_I75_17_05.png](underexposure_I75_17_05.png) `I75_17_05.png` (dist: Darken lvl 5) -- true_label=1, predicted_prob=0.003
- ![underexposure_I52_17_04.png](underexposure_I52_17_04.png) `I52_17_04.png` (dist: Darken lvl 4) -- true_label=1, predicted_prob=0.007
- ![underexposure_I24_17_04.png](underexposure_I24_17_04.png) `I24_17_04.png` (dist: Darken lvl 4) -- true_label=1, predicted_prob=0.008
- ![underexposure_I24_17_05.png](underexposure_I24_17_05.png) `I24_17_05.png` (dist: Darken lvl 5) -- true_label=1, predicted_prob=0.009
- ![underexposure_I24_17_03.png](underexposure_I24_17_03.png) `I24_17_03.png` (dist: Darken lvl 3) -- true_label=1, predicted_prob=0.014
- ![underexposure_I61_17_04.png](underexposure_I61_17_04.png) `I61_17_04.png` (dist: Darken lvl 4) -- true_label=1, predicted_prob=0.015
- ![underexposure_I21_17_03.png](underexposure_I21_17_03.png) `I21_17_03.png` (dist: Darken lvl 3) -- true_label=1, predicted_prob=0.016

## overexposure

- ![overexposure_I52_16_03.png](overexposure_I52_16_03.png) `I52_16_03.png` (dist: Brighten lvl 3) -- true_label=1, predicted_prob=0.003
- ![overexposure_I52_16_04.png](overexposure_I52_16_04.png) `I52_16_04.png` (dist: Brighten lvl 4) -- true_label=1, predicted_prob=0.005
- ![overexposure_I21_16_03.png](overexposure_I21_16_03.png) `I21_16_03.png` (dist: Brighten lvl 3) -- true_label=1, predicted_prob=0.012
- ![overexposure_I24_16_05.png](overexposure_I24_16_05.png) `I24_16_05.png` (dist: Brighten lvl 5) -- true_label=1, predicted_prob=0.015
- ![overexposure_I75_16_03.png](overexposure_I75_16_03.png) `I75_16_03.png` (dist: Brighten lvl 3) -- true_label=1, predicted_prob=0.023
- ![overexposure_I52_16_05.png](overexposure_I52_16_05.png) `I52_16_05.png` (dist: Brighten lvl 5) -- true_label=1, predicted_prob=0.024
- ![overexposure_I24_16_03.png](overexposure_I24_16_03.png) `I24_16_03.png` (dist: Brighten lvl 3) -- true_label=1, predicted_prob=0.031
- ![overexposure_I61_16_03.png](overexposure_I61_16_03.png) `I61_16_03.png` (dist: Brighten lvl 3) -- true_label=1, predicted_prob=0.034

## noise

- ![noise_I52_15_03.png](noise_I52_15_03.png) `I52_15_03.png` (dist: Denoise lvl 3) -- true_label=1, predicted_prob=0.034
- ![noise_I52_15_04.png](noise_I52_15_04.png) `I52_15_04.png` (dist: Denoise lvl 4) -- true_label=1, predicted_prob=0.078
- ![noise_I52_15_05.png](noise_I52_15_05.png) `I52_15_05.png` (dist: Denoise lvl 5) -- true_label=1, predicted_prob=0.088
- ![noise_I21_14_03.png](noise_I21_14_03.png) `I21_14_03.png` (dist: Multiplicative noise lvl 3) -- true_label=1, predicted_prob=0.120
- ![noise_I52_14_03.png](noise_I52_14_03.png) `I52_14_03.png` (dist: Multiplicative noise lvl 3) -- true_label=1, predicted_prob=0.129
- ![noise_I52_11_03.png](noise_I52_11_03.png) `I52_11_03.png` (dist: White noise lvl 3) -- true_label=1, predicted_prob=0.130
- ![noise_I21_13_03.png](noise_I21_13_03.png) `I21_13_03.png` (dist: Impulse noise lvl 3) -- true_label=1, predicted_prob=0.130
- ![noise_I52_13_03.png](noise_I52_13_03.png) `I52_13_03.png` (dist: Impulse noise lvl 3) -- true_label=1, predicted_prob=0.132

## corruption

- ![corruption_I21_18_05.png](corruption_I21_18_05.png) `I21_18_05.png` (dist: Mean shift lvl 5) -- true_label=1, predicted_prob=0.048
- ![corruption_I21_20_05.png](corruption_I21_20_05.png) `I21_20_05.png` (dist: Non-eccentricity patch lvl 5) -- true_label=1, predicted_prob=0.054
- ![corruption_I72_25_05.png](corruption_I72_25_05.png) `I72_25_05.png` (dist: Contrast change lvl 5) -- true_label=1, predicted_prob=0.077
- ![corruption_I21_23_03.png](corruption_I21_23_03.png) `I21_23_03.png` (dist: Color block lvl 3) -- true_label=1, predicted_prob=0.091
- ![corruption_I21_10_03.png](corruption_I21_10_03.png) `I21_10_03.png` (dist: JPEG lvl 3) -- true_label=1, predicted_prob=0.094
- ![corruption_I72_09_03.png](corruption_I72_09_03.png) `I72_09_03.png` (dist: JPEG2000 lvl 3) -- true_label=1, predicted_prob=0.099
- ![corruption_I15_17_05.png](corruption_I15_17_05.png) `I15_17_05.png` (dist: Darken lvl 5) -- true_label=1, predicted_prob=0.105
- ![corruption_I21_06_03.png](corruption_I21_06_03.png) `I21_06_03.png` (dist: Color quantization lvl 3) -- true_label=1, predicted_prob=0.109

## quality_score

- ![quality_score_I52.png](quality_score_I52.png) `I52.png` (dist: none lvl 0) -- true_quality=100.0, predicted_quality=6.8
- ![quality_score_I52_18_02.png](quality_score_I52_18_02.png) `I52_18_02.png` (dist: Mean shift lvl 2) -- true_quality=95.8, predicted_quality=6.7
- ![quality_score_I52_01_01.png](quality_score_I52_01_01.png) `I52_01_01.png` (dist: Gaussian blur lvl 1) -- true_quality=94.2, predicted_quality=6.8
- ![quality_score_I52_03_01.png](quality_score_I52_03_01.png) `I52_03_01.png` (dist: Motion blur lvl 1) -- true_quality=92.5, predicted_quality=6.8
- ![quality_score_I52_08_01.png](quality_score_I52_08_01.png) `I52_08_01.png` (dist: Color saturation 2 lvl 1) -- true_quality=92.5, predicted_quality=6.8
- ![quality_score_I52_16_01.png](quality_score_I52_16_01.png) `I52_16_01.png` (dist: Brighten lvl 1) -- true_quality=91.8, predicted_quality=6.6
- ![quality_score_I52_18_04.png](quality_score_I52_18_04.png) `I52_18_04.png` (dist: Mean shift lvl 4) -- true_quality=93.2, predicted_quality=8.1
- ![quality_score_I52_17_02.png](quality_score_I52_17_02.png) `I52_17_02.png` (dist: Darken lvl 2) -- true_quality=92.5, predicted_quality=7.8
