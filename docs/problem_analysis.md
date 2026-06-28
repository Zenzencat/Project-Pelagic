# Project Pelagic: Problem Analysis Summary
**SWU Prasarnmit — AI Engineering Track (Final Project Checkpoint 1)**

This document outlines the problem analysis, geographical context, and physical principles underlying **Project Pelagic** (SAR-based oil slick detection system).

---

## 1. Problem Statement (ปัญหาและที่มาของโครงการ)

Marine oil pollution is a critical threat to marine ecosystems, coastal tourism, and aquaculture. It originates from two main sources:
1. **Accidental Oil Spills (อุบัติเหตุน้ำมันรั่วไหล)**: Large-scale discharges due to pipeline leaks, offshore platform blowouts, or vessel collisions.
2. **Illegal Bilge Dumping (การลักลอบปล่อยน้ำมันอับเฉาเรือ)**: Small-to-medium discharges of oily bilge water from commercial ships, which occurs frequently and clandestinely to avoid port disposal fees.

### The Southeast Asian Context & Optical Limitations
In Southeast Asia, particularly around the Gulf of Thailand and the Andaman Sea, monitoring oil spills is severely hampered by **monsoon seasons (ฤดูมรสุม)**. 
* Optical satellites (e.g., Sentinel-2, Landsat) capture light in the visible spectrum. Consequently, they are completely blinded by heavy cloud cover, which exceeds 70-80% during rainy seasons.
* Heavy cloud coverage renders optical satellite detection ineffective for days or weeks at a time, allowing illegal dumping to go unnoticed.

---

## 2. The Solution: Synthetic Aperture Radar (SAR)
To overcome the cloud-cover limitation, Project Pelagic leverages **Sentinel-1 C-band Synthetic Aperture Radar (SAR)** imagery.

### 2.1 Why SAR works (ทำไมดาวเทียมเรดาร์จึงสามารถตรวจจับได้):
* **Cloud Penetration (การทะลุทะลวงเมฆ)**: Radar operates in the microwave spectrum (C-band frequency ~5.4 GHz). At this wavelength, radar signals penetrate clouds, rain, fog, and smog, allowing continuous 24/7 monitoring.
* **Active Sensing (ระบบวัดสัญญาณสะท้อนกลับแบบแอกทีฟ)**: Unlike optical sensors that rely on reflected sunlight, SAR satellites emit their own microwave pulses and measure the strength of the backscattered signal (สัญญาณสะท้อนกลับ).

### 2.2 Physics of Oil Slick Visibility (ฟิสิกส์ของการตรวจจับคราบน้ำมัน):
1. **Capillary Waves (คลื่นฝอยผิวหน้าทะเล)**: Clean ocean water has a rough surface due to wind-induced capillary waves. This rough surface scatters radar pulses in all directions, returning a moderate portion back to the satellite (appears as a **grey/bright** background).
2. **Surface Damping (การหน่วงความตึงผิวน้ำ)**: Viscous oil forms a thin layer over the water, increasing surface tension. This dampens the small capillary waves, smoothing out the ocean surface.
3. **Specular Reflection (การสะท้อนแบบกระจก)**: The smooth oil-slick surface acts like a mirror, reflecting the radar energy *away* from the satellite. The sensor receives little to no backscatter return from these areas, causing oil slicks to appear as **dark patches** (พิกเซลสีดำ/มืด) on the SAR image.

---

## 3. Scope and Lookalikes (ขอบเขตและข้อจำกัดการวิเคราะห์)

While SAR is highly effective, it suffers from the problem of **lookalikes (วัตถุลักษณะคล้ายคลึงกัน)**. Dark spots on SAR imagery are not always oil slicks; they can also be caused by:
* **Wind Shadows (พื้นที่อับลม)**: Calm sea zones with no wind do not generate capillary waves, appearing dark.
* **Natural Organic Films (ฟิล์มชีวภาพตามธรรมชาติ)**: Algal blooms or fish oils also dampen waves.
* **Internal Waves (คลื่นใต้น้ำ)**: Disruption of surface currents.

### Evaluation of Lookalike Discrimination (การประเมินเพื่อจำแนกวัตถุคล้ายคลึงกัน)
To ensure the model baseline can distinguish true oil spills from natural lookalikes without high false alarm rates, our held-out test dataset (**Part III of the Trujillo-Acatitla et al. dataset**) contains a strictly balanced split:
* **150 Oil Spill Scenes**: Captures positive targets.
* **150 No Oil Scenes**: Captures clean sea areas.
* **150 Lookalike Scenes**: Captures natural ocean phenomena (wind shadows, organic films, etc.).
Evaluating on this split allows us to calculate precision and recall specifically for lookalike false alarms, proving the model's reliability in academic presentations.

**Vessel Attribution Out of Scope**: Stating definitively which vessel caused a slick is highly complex. It requires hydrodynamic drift modeling, current tracking, and exact spill timelines, which are out of scope for this proof-of-concept. Project Pelagic acts strictly as a **detection screening layer** to flag candidate scenes for human experts.

---

## 4. Thai Terminology Key (ตารางคำศัพท์เทคนิคภาษาไทย)

| English Term | Thai Translation | Description / Context |
| :--- | :--- | :--- |
| **Bilge Dumping** | การปล่อยน้ำมันอับเฉาเรือ | การลักลอบระบายน้ำปนเปื้อนน้ำมันใต้ท้องเรือลงสู่ทะเล |
| **Active Sensor** | เซนเซอร์แบบแอกทีฟ / ส่งสัญญาณวัดเอง | เซนเซอร์ที่ส่งพลังงานออกไปและตรวจจับสัญญาณที่สะท้อนกลับมา |
| **Backscatter Signal** | สัญญาณสะท้อนกลับ | ปริมาณคลื่นเรดาร์ที่สะท้อนจากพื้นผิวโลกกลับมายังตัวรับบนดาวเทียม |
| **Capillary Waves** | คลื่นฝอยผิวหน้าทะเล | คลื่นขนาดเล็กมากบนผิวทะเลที่เป็นตัวทำให้เกิดการสะท้อนกลับของเรดาร์ |
| **Surface Damping** | การลดแรงตึงผิวและหน่วงคลื่น | ปรากฏการณ์ที่คราบน้ำมันไปกดคลื่นฝอยทำให้ทะเลราบเรียบขึ้น |
| **Specular Reflection** | การสะท้อนแบบกระจกเงา | การที่คลื่นเรดาร์ตกกระทบผิวเรียบแล้วสะท้อนสะเปะสะปะออกไปทิศอื่น |
| **Lookalike** | วัตถุลักษณะคล้ายคราบน้ำมัน | ปรากฏการณ์ธรรมชาติอื่น ๆ เช่น พื้นที่อับลม หรือ สาหร่ายสะสมที่ทำให้ภาพมืด |
| **Spatial Resolution** | ความละเอียดเชิงพื้นที่ | ขนาดพื้นที่จริงบนโลกที่แทนด้วย 1 พิกเซลในภาพถ่ายดาวเทียม |
