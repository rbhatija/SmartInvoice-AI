📄 SmartInvoice Text Detection (DBNet) – README

This project implements a DBNet-based text detection pipeline for invoice images using PyTorch and a pretrained ResNet18 backbone. The system converts OCR polygon annotations into segmentation masks, trains DBNet to detect text regions, and visualizes predictions on sample invoices.

Features Implemented

Load invoice dataset from HuggingFace (invoices-and-receipts_ocr_v1).

Parse OCR box annotations and convert them to DBNet-friendly polygons.

Generate full-resolution text masks from polygons.

Build a custom DBNetDataset with correct scaling + preprocessing.

Implement DBNet with ResNet18 backbone and segmentation head.

Train the model with upsampled logits and full-resolution masks.

Save trained model weights to local storage + Google Drive.

Visualize predictions:

  Input image
  
  Ground-truth mask
  
  Predicted probability heatmap
  
  Overlay of prediction on image
