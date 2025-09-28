# QCT medical image auto annotation for femur bone Dataset


---
**Table of content**



---
---
**Steps**
1. Data processing
    A. Manual annotation
    B. DataSet creation
---

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

32 has serious issue

============================================================
ANALYSIS SUMMARY
============================================================
Image Volume: (256, 512, 177) (float64)
Value Range: [-1024.0000, 3071.0000]
Mean ± Std: -795.9629 ± 422.1146

Label Volume: (256, 512, 177) (int64)
Unique Labels: [np.int64(0), np.int64(1), np.int64(2), np.int64(3)]
Overall Annotation Coverage: 13.62% of voxels

Slice Coverage: 177/177 slices have annotations (100.00%)
Average annotation per annotated slice: 13.62%

Label Distribution (entire volume):
  Background: 20,039,628 v oxels (86.38%)
  Annotation 1: 88,598 voxels (0.38%)
  Annotation 2: 190,971 voxels (0.82%)
  Annotation 3: 2,880,547 voxels (12.42%)

Detailed analysis saved to: comprehensive_analysis.txt

Showing 3 sample slices with annotations:
  Slice 138: 18.87% annotated
  Slice 71: 15.18% annotated
  Slice 0: 1.08% annotated





  ============================================================
ANALYSIS SUMMARY
============================================================
Image Volume: (256, 512, 177) (float64)
Value Range: [-1024.0000, 3071.0000]
Mean ± Std: -795.9629 ± 422.1146

Label Volume: (256, 512, 182) (float64)
Unique Labels: [np.int64(0), np.int64(1), np.int64(2)]
Overall Annotation Coverage: 3.34% of voxels

Slice Coverage: 177/177 slices have annotations (100.00%)
Average annotation per annotated slice: 3.34%

Label Distribution (entire volume):
  Background: 23,057,179 voxels (96.66%)
  Annotation 1: 160,749 voxels (0.67%)
  Annotation 2: 637,176 voxels (2.67%)

Detailed analysis saved to: comprehensive_analysis.txt

Showing 3 sample slices with annotations:
  Slice 153: 5.36% annotated
  Slice 79: 3.78% annotated
  Slice 0: 1.25% annotated