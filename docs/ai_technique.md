# Project Pelagic: AI Technique Justification
**SWU Prasarnmit — AI Engineering Track (Final Project Checkpoint 1)**

This document explains and justifies the choice of Machine Learning algorithms, dataset strategy, and evaluation metrics for **Project Pelagic**.

---

## 1. Selected ML Task: Binary Semantic Segmentation
The system models oil slick detection as a **Binary Semantic Segmentation (การแบ่งส่วนภาพเชิงความหมายแบบสองส่วน)** task. 
* **Input**: A 2-channel tensor of dimension `(2, H, W)` representing normalized `(VV, VH)` polarization radar backscatter.
* **Output**: A binary mask of dimension `(1, H, W)` where a pixel value of `1` represents an oil slick and `0` represents the background (clean sea, land, or lookalikes).

---

## 2. Model Architecture: U-Net (ResNet-34 Encoder)
We select **U-Net** as our baseline model architecture.

```
Encoder (Downsampling)              Decoder (Upsampling)
[Conv Block] --------------------> [Up-Conv + Skip Connection]
   [Pooling]                         [Transpose Conv]
      \                                     ^
       v                                   /
     [Conv Block] --------------> [Up-Conv + Skip Connection]
        [Pooling]                   [Transpose Conv]
           \                             ^
            v                           /
          [Conv Block] -------> [Up-Conv + Skip Connection]
             [Pooling]             [Transpose Conv]
                \                       ^
                 v                     /
                 [ Bottleneck Block ]
```

### Why U-Net is Selected:
1. **Preserving Spatial Details via Skip Connections (การรักษาขอบเขตพิกเซลผ่านโครงข่ายข้ามสาย)**:
   * During downsampling (encoder), the spatial resolution decreases as high-level semantic features are extracted.
   * U-Net uses **skip connections** to copy fine-grained spatial feature maps from the encoder directly to the decoder. This enables the model to reconstruct highly precise oil slick boundaries and narrow linear slick shapes.
2. **Data Efficiency**:
   * U-Net is highly efficient at training on small datasets (~1,000 labeled images) compared to vision transformers like SegFormer, which require tens of thousands of images to generalize without overfitting.
3. **Compute Constraints (ข้อจำกัดด้านกำลังการคำนวณ)**:
   * U-Net is lightweight and can run inference easily on a standard CPU, ensuring that the FastAPI backend can process frames within seconds without requiring expensive GPU hosting.

---

## 3. Addressing Extreme Class Imbalance (การแก้ปัญหาความไม่สมดุลของข้อมูล)
In typical Sentinel-1 ocean scenes, oil slick pixels cover less than 1% of the total area. 
* If we use standard **Binary Cross Entropy (BCE) Loss**, the model will achieve 99% accuracy simply by predicting all pixels as "background" (0).
* To resolve this, we employ a **Joint Loss Function** combining BCE and **Dice Loss**:

$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{BCE}} + \beta \mathcal{L}_{\text{Dice}}$$

* **Dice Loss**: Directly optimizes for the overlap (Intersection over Union) between the predicted slick and ground truth, ignoring background pixel counts.
* **Focal Loss (Alternative)**: Down-weights the loss of easy-to-classify background pixels and focuses the training signal on the rare, hard-to-classify slick boundaries.

---

## 4. Evaluation Metrics (ดัชนีวัดผลสัมฤทธิ์ของโมเดล)
We explicitly reject overall pixel accuracy as a metric. We report:
1. **Intersection over Union (IoU) / Jaccard Index**: The size of the intersection divided by the size of the union of predicted and ground-truth slick pixels.
2. **F1-Score / Dice Coefficient**: The harmonic mean of Precision and Recall for the oil slick class.
3. **Precision & Recall**: 
   * **Precision** measures the ratio of true slicks detected out of all flagged pixels (reduces false alarms from lookalikes).
   * **Recall** measures the ratio of true slicks detected out of all actual slicks (minimizes missed detections).

---

## 5. Dataset Splitting & Preprocessing Discipline (การจัดสรรชุดข้อมูลและการเตรียมประมวลผล)

We utilize the three-part Sentinel-1 SAR Oil Spill Image Dataset by Trujillo-Acatitla et al. containing 2048x2048px dual-channel (VV, VH) images in dB scale:
1. **Model Training & Hyperparameter Tuning**:
   * We combine **Part I** (1,200 oil spill scenes) and **Part II** (1,370 clean sea and lookalike scenes: 685 of each category) to build our training and validation datasets (totaling 2,570 scenes).
   * Cross-validation splits are partitioned strictly by Scene ID to prevent spatial leakage.
2. **Model Evaluation & Reporting**:
   * We keep **Part III** (450 scenes: 150 oil, 150 no oil, 150 lookalike) completely untouched during model training and tuning.
   * All reported metrics (IoU, Dice coefficient, Precision, Recall) are measured and reported exclusively on this held-out Part III test set to ensure academic integrity.
3. **Coordinate Reference System (CRS) Alignment**:
   * **Important Constraint**: The binary segmentation masks provided in the dataset are plain pixel matrices and do not contain georeferenced coordinate metadata.
   * Consequently, the data pipeline treats masks as plain numpy matrices. CRS coordinate projection and spatial mapping are handled strictly on the source satellite images, eliminating complex CRS mask-alignment overhead.

---

## 6. Thai Terminology Key (ตารางคำศัพท์เทคนิคภาษาไทย)

| English Term | Thai Translation | Description / Context |
| :--- | :--- | :--- |
| **Semantic Segmentation** | การแบ่งส่วนภาพเชิงความหมาย | การคัดแยกและระบุระดับคลาสให้กับภาพในระดับรายพิกเซล |
| **Encoder / Decoder** | โครงสร้างเข้ารหัส / ถอดรหัส | ตัวดึงคุณลักษณะสำคัญจากภาพ (Encoder) และตัวสร้างภาพความละเอียดเท่าเดิมกลับขึ้นมา (Decoder) |
| **Skip Connection** | การเชื่อมโยงทางข้าม | การส่งผ่านข้อมูลลักษณะเด่นจากช่วงแรกของโมเดลไปยังช่วงท้ายเพื่อรักษารายละเอียดขอบภาพ |
| **Class Imbalance** | ความไม่สมดุลของคลาสข้อมูล | ปัญหาที่สัดส่วนระหว่างสิ่งที่ต้องการตรวจหา (คราบน้ำมัน) มีน้อยมากเมื่อเทียบกับพื้นหลัง (ทะเล) |
| **Dice Loss** | ฟังก์ชันการสูญเสียแบบไดซ์ | ฟังก์ชันคำนวณความสูญเสียโดยวัดจากเปอร์เซ็นต์พื้นที่ที่ซ้อนทับกันระหว่างผลลัพธ์กับเฉลย |
| **Focal Loss** | ฟังก์ชันการสูญเสียโฟคัล | ฟังก์ชันความสูญเสียที่เน้นการปรับค่าน้ำหนักให้กับพิกเซลที่เรียนรู้ได้ยาก |
| **Intersection over Union (IoU)** | ดัชนีวัดการซ้อนทับกันของพื้นที่ | อัตราส่วนพื้นที่ส่วนที่ซ้อนทับกัน (อินเตอร์เซกชัน) หารด้วยพื้นที่รวม (ยูเนียน) |
| **Overfitting** | การเรียนรู้จำเพาะเกินไป | ปัญหาที่โมเดลจดจำข้อมูลชุดฝึกสอนได้ดีเยี่ยม แต่ไม่สามารถประมวลผลข้อมูลใหม่ได้ดี |
