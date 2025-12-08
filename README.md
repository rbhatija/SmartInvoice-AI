# SmartInvoice AI
## DBNet-Inspired Text Region Segmentation for Invoices and Receipts

SmartInvoice AI is a deep learning system designed to automatically detect text regions in financial documents using a DBNet-style semantic segmentation model with a ResNet-18 backbone. The model identifies pixel-level text areas from OCR-annotated polygons, enabling downstream OCR and automated invoice processing workflows.

## Key Features:

* DBNet-inspired segmentation architecture
* ResNet-18 pretrained backbone
* Pixel-level text region detection
* Polygon-to-mask preprocessing pipeline
* Threshold sweep evaluation (0.10–0.90)
* High-quality visualizations (heatmaps, overlays, bounding boxes)
* Suitable for OCR and document understanding systems

## Dataset

The project uses the Invoices-and-Receipts OCR Dataset:
**Source:** HuggingFace
**URL:** https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v1

* Training images: 1634
* Validation images: 409
* Labels: OCR polygons converted into segmentation masks
* Image types: Receipts, invoices, store printouts, mixed layouts

## Model Architecture

SmartInvoice AI implements a lightweight DBNet-style architecture:
* Backbone: ResNet-18 (convolutional layers only)
* Head:
  * 1×1 Conv: 512 → 64
  * 3×3 Conv: 64 → 1
* Output: Text probability map
* Loss: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)

## Performance Summary

**Average IoU:** 0.6901  
**F1-Score (0.50 threshold):** 0.7816  
**Best F1 Threshold Range:** 0.30–0.40 (≈0.79)  

Threshold sweep analysis demonstrates predictable precision–recall trade-offs for configurable OCR pipelines.

