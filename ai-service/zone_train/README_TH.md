# โมเดลวิเคราะห์ผิวรายโซน (Zone classifier)

ขั้นตอนนี้แบ่งใบหน้าเป็น 5 ส่วน ได้แก่ **หน้าผาก, แก้ม 2 ข้าง, จมูก, ใต้ตา, คาง** แล้วให้โมเดลวิเคราะห์ผิวทีละส่วน
ส่วนละ 6 ปัญหา คือ สิว จุดด่างดำ ถุงใต้ตา ความมัน รอยแดง และริ้วรอย (multi-label classification)
เป้าหมายคือวัดผลด้วย **Accuracy / F1** ให้ได้ 80–90%

```
ภาพใบหน้า ─► MediaPipe FaceMesh (468 จุด) ─► 6 polygon ─► ครอปทีละโซน 224×224 ─► CNN + zone embedding ─► 6 ความน่าจะเป็น/โซน
```

| ไฟล์ | หน้าที่ |
|---|---|
| `ai-service/face_zones.py` | แบ่งโซนจาก landmark และครอปภาพ ใช้โค้ดชุดเดียวกันทั้งตอนเทรนและใน API |
| `ai-service/zone_model.py` | โครงสร้างโมเดลและตัวโหลดสำหรับ API |
| `zone_train/build_zone_dataset.py` | แปลง dataset กรอบ YOLO (`dataset_clean_v2`) เป็น dataset รายโซน |
| `zone_train/train_zone_classifier.py` | เทรนโมเดล ปรับ threshold บน valid และวัดผลบน test (Accuracy, Precision, Recall, F1, AUC) |

---

## 1. ติดตั้ง (ครั้งเดียว บนเครื่องที่มี RTX 3070)

```bat
cd "E:\Project END\Sangmai\skinglow-trian"
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mediapipe==0.10.21 "numpy<2" opencv-python pillow matplotlib
python -c "import torch; print(torch.cuda.is_available())"
```

บรรทัดสุดท้ายต้องขึ้น `True` ถ้าขึ้น `False` แปลว่า torch ที่ติดตั้งไม่รองรับ CUDA ให้ติดตั้งใหม่จาก index `cu121` ตามบรรทัดแรก

## 2. สร้าง dataset รายโซน (ประมาณ 5–10 นาที)

```bat
set REPO=E:\path\to\Skinglow
python "%REPO%\ai-service\zone_train\build_zone_dataset.py" --data-root "E:\Project END\Sangmai\skinglow-trian\dataset_clean_v2" --out zone_dataset
```

เมื่อรันเสร็จจะได้ไฟล์เหล่านี้:
- `zone_dataset\labels.csv` เก็บ 1 แถวต่อ 1 ครอป พร้อม label ของ 6 คลาส (1 = มี, 0 = ไม่มี, -1 = ไม่ทราบ)
- `zone_dataset\stats.json` เก็บจำนวน positive/negative/unknown ต่อคลาส และกฎที่ใช้ ใส่ในบท methodology ได้เลย
- `zone_dataset\preview\` เก็บภาพที่วาดโซนทับกับกรอบเดิม **ควรเปิดดูสัก 10 ภาพก่อนเทรน**

### ทำไมต้องมี label "-1 ไม่ทราบ"
`dataset_clean_v2` รวมมาจากหลายแหล่ง และแต่ละแหล่ง label ไว้ไม่ครบทุกคลาส ตัวอย่างเช่น
- ชุด `LINE_ALBUM_oily-skin` มีแค่ oiliness
- ชุด `levle*` มีแค่ black_spot
- ชุด `img-*` มีแค่ wrinkle

ถ้านับกรณี "ไม่มีกรอบสิว" ในชุดเหล่านี้ว่าเป็น "ไม่มีสิว" เราจะสอนโมเดลผิด และได้ค่าความแม่นยำที่ไม่จริง
สคริปต์จึงตั้งค่าเป็น -1 แล้วไม่นำไปคิดทั้งใน loss และใน metric กรณีที่ได้ -1 มีดังนี้
- คลาสที่แหล่งข้อมูลนั้นไม่เคย label
- กรอบใหญ่ที่แตะโซนเพียงนิดเดียว
- ภาพที่ไม่มี label เลย เพราะตรวจแล้วพบภาพที่เห็นสิว/จุดชัดแต่ไม่มีกรอบ

ภาพซูมใกล้ที่ไม่เห็นทั้งใบหน้ามีประมาณ 1/4 ของ dataset ภาพกลุ่มนี้แบ่งโซนไม่ได้ สคริปต์จึงเก็บไว้เป็นตัวอย่างชนิด `patch`
เพื่อใช้ช่วยสอนลักษณะของสิว/จุดในระยะใกล้ แต่แยกรายงานผลออกจากคะแนนรายโซน

## 3. เทรน + วัดผล (RTX 3070 ประมาณ 20–40 นาที)

```bat
python "%REPO%\ai-service\zone_train\train_zone_classifier.py" --data zone_dataset --arch efficientnet_b0 --epochs 40 --batch 64 --workers 4
```

ผลลัพธ์จะอยู่ใน `runs_zone\<ชื่อรัน>\`

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| `report_test.md` | **ตารางผลบน test set สำหรับใส่ในเล่ม** มี Macro-F1, Accuracy, Precision, Recall, F1, AUC ต่อคลาส และ Accuracy/F1 ต่อโซน |
| `metrics_test.json` | ตัวเลขชุดเดียวกันในรูป JSON |
| `curves.png` | กราฟ loss และ F1 ต่อ epoch |
| `best.pt` | ไฟล์โมเดลที่จะนำไป deploy |

**ควรรายงานตัวเลขไหน:** ใช้ **Macro-F1** เป็นตัวเลขหลัก และรายงานคู่กับ Balanced accuracy
Accuracy อย่างเดียวจะสูงเกินจริงเสมอ เพราะโซนส่วนใหญ่ไม่มีปัญหา โมเดลที่ตอบว่า "ไม่มีสิว" ทุกโซนก็ได้ accuracy สูงแล้ว
ส่วน threshold ปรับบน valid แล้วใช้ค่าเดิมกับ test จึงไม่ได้ปรับให้เข้ากับ test set

### ถ้า Macro-F1 ยังไม่ถึง 80%
ลองทีละข้อ แล้วเทียบกันด้วย `report_test.md`
1. ใช้โมเดลใหญ่ขึ้นและภาพละเอียดขึ้น: `--arch efficientnet_b2 --img 288 --batch 32`
2. ดูในตาราง per-class ว่าคลาสไหนต่ำ ปกติจะเป็น redness และ eyebag ซึ่งมีข้อมูลน้อยที่สุด
   ทางแก้ที่ได้ผลที่สุดคือเพิ่มภาพหรือ label ของคลาสนั้น การปรับพารามิเตอร์ช่วยได้น้อยกว่า
3. ถ้าต้องการสอนโมเดลว่าผิวใสเป็นอย่างไร และแน่ใจว่าภาพที่ไม่มี label เป็นผิวใสจริง ให้สร้าง dataset ใหม่ด้วย `--trust-empty-images`
4. label ใหม่เฉพาะภาพหน้าตรงให้ครบทั้ง 6 คลาส สัก 300–500 ภาพ (ใช้ Roboflow/CVAT) จะช่วยลด label "ไม่ทราบ" ได้มาก

## 4. นำไปใช้ในแอป

```bat
copy runs_zone\<ชื่อรัน>\best.pt "%REPO%\ai-service\models\zone_classifier.pt"
```

- `main.py` จะโหลดไฟล์นี้อัตโนมัติ ตรวจสอบได้ที่ `/health` ซึ่งต้องขึ้น `"zone_model_type": "zone_classifier"`
- ถ้าไม่มีไฟล์นี้ API ยังแบ่งโซนและให้คะแนนรายโซนได้ โดยนับจากกรอบ YOLO ที่ตกในแต่ละโซน (`yolo_per_zone`) รูปแบบ response เหมือนเดิมทุกประการ
- ไฟล์โมเดลมีขนาดประมาณ 16 MB (efficientnet_b0) และใช้เวลาประมาณ 0.2 วินาทีต่อภาพ (6 โซน) บน CPU 1 core

## 5. สิ่งที่เปลี่ยนในแอป

- **API (`/analyze`)** มีฟิลด์ใหม่ `zones` ซึ่งแต่ละโซนมี `zone`, `name_th`, `score` (0–100), `issues` (ปัญหาและความน่าจะเป็น), `polygons`, `box`
  - `skin_score` เปลี่ยนเป็นค่าเฉลี่ยของคะแนนทุกโซน
  - รับฟิลด์ใหม่ `mirrored` สำหรับภาพจากกล้องหน้าที่ถูกกลับด้าน เพื่อให้ระบุแก้มซ้าย/ขวาถูกต้อง
- **PHP**
  - `analyze.php` บันทึก `zones_json` และสรุปผลเป็นภาษาไทยรายโซน เช่น "หน้าผาก: ริ้วรอย · แก้มซ้าย: สิว"
  - `BoundingBoxDrawer` วาดเส้นขอบโซนลงบนภาพ
  - `history.php` ส่ง `zones` กลับไปด้วย
- **ฐานข้อมูล** ต้องรัน `database/migration_add_zones_json.sql` หนึ่งครั้ง ถ้ายังไม่รัน ระบบยังทำงานได้ แต่จะไม่บันทึกผลรายโซน
- **หน้าเว็บ** มีการ์ดผลรายโซน 6 ใบ แสดงคะแนน แถบสี และปัญหาที่พบพร้อม %
