# Project Pelagic: UX/UI Wireframe Design
**SWU Prasarnmit — AI Engineering Track (Final Project Checkpoint 1)**

This document details the user interface (UI) and user experience (UX) layout design for the **Project Pelagic** map-based operator dashboard.

---

## 1. Interactive Wireframe Link (ลิงก์เข้าชมหน้าจอตัวอย่าง)

We have created an interactive HTML wireframe. You can open it directly in your web browser to test the layout, select historical oil slick detections, toggle map layers, and run a mock model detection simulation:

👉 **[Open Interactive HTML Wireframe (docs/wireframe.html)](file:///c:/Users/Iris/Downloads/Project%20Pelagic/docs/wireframe.html)**

---

## 2. Layout Structure Design (โครงสร้างเลย์เอาต์การจัดวางหน้าจอ)

The interface is structured into a modern split-pane design using a dark-theme color palette. Dark themes reduce operator eye strain during night shifts and allow bright, glowing detection overlays to stand out clearly.

```
+------------------------------------------+-----------------------+
|  PROJECT PELAGIC                         | [ ] Footprint layer   |
|  (SAR Oil Slick Detection)               | [x] Oil Slick layer   |
|                                          | [x] AIS Vessel layer  |
|  [ Recent Detections List ]              |                       |
|  +-------------------------------------+ |   (Interactive Map)   |
|  | Scene: S1A_IW_GRDH_1SDV_20260627... | |                       |
|  | Area: 14.5 sq km  | Conf: 94.2%     | |      +--------+       |
|  +-------------------------------------+ |      | Slick  |       |
|  | Scene: S1B_IW_GRDH_1SDV_20260625... | |      | Polygon|       |
|  +-------------------------------------+ |      +--------+       |
|                                          |         :             |
|  [ Detection Metadata Details ]          |         : (dist)      |
|  Coordinates: 9.200°N, 101.500°E         |         :             |
|  Area: 14.5 sq km                        |      (o) AIS Vessel   |
|  AIS Vessels Nearby: 3                   |                       |
|  +-------------------------------------+ |                       |
|  |  SIMULATE NEW SAR DETECTION         | |                       |
|  +-------------------------------------+ |                       |
+------------------------------------------+-----------------------+
```

### 2.1 Sidebar Panel (แถบเมนูด้านซ้าย)
* **Header & Brand Logo**: Displays "Project Pelagic" with a subtle pulse animation on the logo, indicating the system is actively monitoring.
* **Recent Detections List**: Displays list cards of processed SAR scenes in chronological order. Each card summarizes the Scene ID, model confidence score, slick area size, and number of correlated AIS vessels. Clicking on a card updates the map and detail panels.
* **Detection Metadata Details**: Displays precise coordinate centroids, timestamp, exact calculated area size, and vessel counts for the selected detection.
* **Simulation Trigger**: A button to simulate running a new Sentinel-1 scene through the U-Net model. It triggers a processing loader and notifies the user upon successfully rendering the new slick.

### 2.2 Map Display Panel (พื้นที่แผนที่เชิงตอบโต้)
* **Dark Matter Basemap**: Uses dark map tiles to provide high contrast against the bright, semi-translucent detection overlays.
* **Scene Footprint**: Indicated by a blue dashed rectangle, outlining the boundaries of the Sentinel-1 radar capture.
* **Oil Slick Polygon**: Indicated by a glowing, semi-translucent magenta polygon showing the exact pixels segmented by the U-Net model. Hovering over the polygon displays an area tooltip.
* **AIS Telemetry Overlay**: Vessels are plotted as orange circular markers. A dotted line links each vessel to the oil slick center, showing distance calculations.

### 2.3 Layer Control (การควบคุมเลเยอร์แผงแผนที่)
* A floating glassmorphism card in the top-right allows operators to toggle map layers on/off dynamically (Footprint, Slick Mask, and AIS Vessel overlays).

---

## 3. Thai Terminology Key (ตารางคำศัพท์เทคนิคภาษาไทย)

| English Term | Thai Translation | Description / Context |
| :--- | :--- | :--- |
| **Interactive Wireframe** | โครงร่างหน้าจอแบบโต้ตอบได้ | ตัวอย่างหน้าจอจำลองที่ผู้ใช้งานสามารถคลิกเลือกปุ่มและดูผลการทำงานได้ |
| **Operator Dashboard** | แผงควบคุมสำหรับผู้ปฏิบัติการ | หน้าจอศูนย์กลางสำหรับเจ้าหน้าที่เพื่อติดตามและตรวจสอบการตรวจจับคราบนํ้ามัน |
| **Dark Theme** | ชุดรูปแบบสีเข้ม / โหมดมืด | การปรับแต่งสีของระบบให้เป็นสีทึบเพื่อถนอมสายตาและช่วยขับเน้นข้อมูลที่ตรวจจับ |
| **Map Layer** | ชั้นข้อมูลแผนที่ | การแยกข้อมูลออกเป็นชั้น ๆ เพื่อความง่ายในการเลือกแสดงผล (เช่น เลเยอร์คราบน้ำมัน หรือพิกัดเรือ) |
| **Vessel Marker** | เครื่องหมายระบุตำแหน่งเรือ | หมุดพิกัดหรือวงกลมที่ใช้วางบนแผนที่เพื่อระบุตำแหน่งของเรือที่ดึงมาจากข้อมูล AIS |
| **Simulation Loader** | ตัวแสดงสถานะโหลดการจำลอง | ตัวหมุนโหลดที่แสดงว่าโมเดลกำลังประมวลผลข้อมูลใหม่ เพื่อให้ระบบเสมือนมีการทำงานจริง |
| **Tooltip** | กล่องข้อความแนะนำพิกัด | ข้อความขนาดเล็กที่ปรากฏขึ้นเมื่อนำเมาส์ไปชี้ที่จุดพิกัดหรือรูปภาพ |
