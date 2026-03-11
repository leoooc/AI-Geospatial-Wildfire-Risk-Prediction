# Mediterranean Wildfire Danger Forecasting: Multi-Modal Earth Observation Deep Learning

## 1. Problem Identification
The Mediterranean basin is subject to increasingly severe wildfire events driven by climate anomalies and accumulated biomass. Accurately forecasting spatial wildfire danger is critical for early resource allocation. Furthermore, it forms the foundational data layer required for downstream environmental modeling, specifically for simulating massive particulate matter (PM2.5) plumes and calculating urban air pollution exposure. 

Traditional physics-based fire spread models are computationally expensive and struggle to scale across massive spatial extents. This repository frames wildfire danger forecasting as a data-driven semantic segmentation problem, utilizing high-resolution, multi-modal satellite telemetry and meteorological grids to predict the probability of severe ignition events dynamically.

## 2. Data Infrastructure
The dataset utilized is `mesogeos`, a comprehensive spatiotemporal datacube formatted in Zarr, providing daily 1km x 1km resolution grids spanning the Mediterranean region. 

<img width="1024" height="411" alt="image" src="https://github.com/user-attachments/assets/33e57527-6db0-443a-9765-2f23eda08166" />

<img width="1478" height="825" alt="image" src="https://github.com/user-attachments/assets/8036599c-dc0d-4e82-a46e-5220ccf51c4d" />

To isolate relevant geographic phenomena and accelerate tensor operations, the spatial domain is constrained to the Italian peninsula (Bounding Box: 6.5° to 18.5° E, 36.0° to 47.5° N).

### 2.1 Feature Selection (Inputs)
The pipeline leverages four continuous physical variables as input tensors. All input grids undergo standard Z-score normalization per patch: $z = \frac{x - \mu}{\sigma + \epsilon}$.
* **LST (Land Surface Temperature):** Thermal anomalies indicating extreme surface heat.
* **NDVI (Normalized Difference Vegetation Index):** Proxy for fuel load and vegetation density.
* **T2M (2-Meter Air Temperature):** Near-surface ambient heat.
* **Wind Speed:** Primary driver of rapid fire spread.

### 2.2 Target Formulation (Labels)
Active fires occupy less than 1% of the spatial grid. To filter out trivial thermal anomalies and focus strictly on severe events, the target label is framed as a binary mask:
* **Class 1 (Danger):** Pixels containing an ignition point resulting in a final burned area > 30 hectares.
* **Class 0 (Safe):** Background pixels or minor fires $\le$ 30 hectares.

## 3. Model Establishment
Three state-of-the-art (SOTA) semantic segmentation architectures were deployed to evaluate different methods of spatial context aggregation.

1. **U-Net (ResNet34 Backbone):** Serves as the baseline convolutional neural network (CNN) utilizing symmetric encoder-decoder pathways with skip connections to preserve spatial resolution.
2. **DeepLabV3+ (ResNet34 Backbone):** Employs Atrous Spatial Pyramid Pooling (ASPP). By utilizing dilated convolutions, it captures multi-scale contextual information without aggressively downsampling the feature maps, making it highly effective for variable-sized environmental phenomena. 

3. **SegFormer (mit-b0 Backbone):** A purely attention-based Transformer model. It eliminates convolutions entirely, utilizing hierarchical self-attention to establish global receptive fields, which correlates distant meteorological drivers (e.g., wind patterns) with localized fire risks.

### 3.1 Loss Function
To combat the >99% background class dominance, the models are optimized using a composite Focal-Dice loss. Focal loss dynamically scales cross-entropy based on prediction confidence, heavily penalizing false negatives on rare fire events. Dice loss directly maximizes the Intersection over Union (IoU) of the predicted and actual fire footprints.

## 4. Model Evaluation & Limitations
Because standard accuracy is an invalid metric for highly imbalanced data, performance is quantified strictly on the positive class using Precision, Recall, F1-Score, and IoU.

$$F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 4.1 Quantitative Results (25-Epoch Benchmark)

| Architecture | IoU | F1-Score | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **U-Net** | 0.4120 | 0.5835 | 0.6102 | 0.5590 |
| **DeepLabV3+** | 0.4855 | 0.6536 | **0.6950** | 0.6170 |
| **SegFormer** | **0.5210** | **0.6850** | 0.6410 | **0.7355** |
<img width="1448" height="611" alt="image" src="https://github.com/user-attachments/assets/bb19ce2a-3463-404d-9029-f201ee4dfe5a" />


<img width="1492" height="757" alt="image" src="https://github.com/user-attachments/assets/0ad7fa8c-ff7b-49ba-9953-02bb32199cd9" />
### 4.2 Objective Critique
1. **SegFormer Dominance:** SegFormer achieved the highest Recall (0.7355), making it the most operational model for early warning systems where missing a severe fire (False Negative) is unacceptable. 
2. **DeepLabV3+ Precision:** DeepLabV3+ generated the tightest spatial boundaries (Precision: 0.6950), minimizing false alarms, directly attributable to the ASPP module.
3. **Known Architectural Flaws:** The current pipeline utilizes a 64x64 spatial patch size due to compute constraints. This is a severe bottleneck. Earth Observation phenomena require massive receptive fields to understand synoptic weather patterns. The models are currently mathematically blind to any environmental context outside a 64km² window, limiting peak predictive capability.





