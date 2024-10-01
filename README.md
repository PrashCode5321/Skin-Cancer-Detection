# Skin Cancer Detection Using Image Processing

## Introduction

Skin cancer is the most common form of cancer, accounting for nearly 40% of all cancers. Early diagnosis is crucial in curbing this devastating disease. This project aims to detect skin cancer at an early stage using image processing techniques.

### Objective

Our main goal is to input an image of a skin lesion and accurately locate, segment, and identify it as a cancer cell using image processing techniques like contrast enhancement, thresholding, color processing, and image segmentation.

### Challenges

- Huge diversity of input images
- Non-uniform illumination, low-lighting, and noisy images
- Low resolution and poor quality images from smartphones
- Diverse range of skin complexions
- Obstructions such as hair
- Similarity in color between skin lesions and surrounding skin

## Methodology

Our approach involves the following steps:

1. Input image resizing to 400x400px
2. Contrast enhancement using CLAHE algorithm
3. Color space conversion from RGB to LAB
4. Applying CLAHE to the L channel
5. Converting back to RGB color space
6. Morphological operations to remove obstructions like noise and hair
7. Inverse binary thresholding
8. Contour segmentation to identify the skin lesion
9. Extraction of the segmented skin lesion

## Implementation

We used Python, OpenCV, and Numpy to implement our solution. The key steps include:

1. CLAHE algorithm for contrast enhancement
2. Color space conversion between RGB and LAB
3. Morphological operations for noise and hair removal
4. Contour segmentation for identifying the skin lesion
5. Bitwise operations to extract the segmented lesion

## Results and Discussion

Our implementation successfully extracted skin lesions from various input images, including those with different sizes, illumination conditions, and obstructions like hair. The output images show properly extracted skin lesions, clear of obstructions, with good contrast and color.

Some areas for improvement include:
- Handling darker hair colors during thresholding
- Improving image quality enhancement techniques for low-quality input images

## Conclusion

This study demonstrates the power of fundamental image processing techniques in medical image analysis. We successfully segmented skin lesions from input images using a combination of contrast enhancement, morphological operations, and contour segmentation.

## Future Work

1. Fine-tuning the CNN used for testing enhanced images
2. Exploring more sophisticated neural network architectures like ResNet
3. Developing custom feature extraction techniques
4. Improving hair removal techniques without blurring the image

## Contributors

- Arvin Das (190929016): Algorithm investigation, documentation
- Anirudh Bharadwaj (190929118): Literature collection and research, documentation
- Prashant (190929114): Workflow design, algorithm investigation, implementation
- Rishabh Dugar (190929056): Workflow design, implementation