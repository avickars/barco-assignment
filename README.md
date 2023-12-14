# Barco Interview Assignment

The following code base is for the program [main.cpp](./main.cpp) that contains a command line application that either (1) applies a user defined tint to an image or (2) Performs
a colour space transformation from BT.709 to BT.2020.  The transformations are implemented using CUDA where the program (1) Reads the image using OpenCV (2) Copies the Image to GPU Memory using CUDA
(3) Applies a [transformation](#transformations) and (4) Copies the image back to host memory and writes the result to a file.

## Notes
- The Transformations are intentionally implemented using separate CUDA operations to maximize readability.  I do recognize that the operations could be merged to be more efficient.  For instance, in the Tint
transformation, the channel conversion from BGR to RGB, the tint operation and the operation from RGB to BGR could be merged into a single operation. These are intentionally left as separate operations so that the transformations
are well-defined for easy readability.
- Aside from the above point, the program could be improved by taking advantage of TensorCores. The ability of TensorCores to significantly improve computation speed.  Furthermore, some gains 
could be made by optimizing the kernal execute for a specific GPU as well.

## Transformations

### Tint
Defined in the [include/Tint_Transformation.h](./include/Tint_Transformation.h) / [include/Tint_Transformation.cpp](./src/Tint_Transformation.cpp) files, the program
applies a user defined tint with an inputted weight to the input image.

### Colour Space Transformation

Defined in the [include/Colour_Space_Transformation.h](./include/Colour_Space_Transformation.h) / [include/Colour_Space_Transformation.cpp](./src/Colour_Space_Transformation.cpp) files, the program
converts the image from BT.709 to BT.2020 colour spaces. The transformation matrix is statically defined in [include/Colour_Space_Transformation.cpp](./src/Colour_Space_Transformation.cpp) (I pre-computed it myself in Python).

## Building 

## Dependencies
1. OpenCV - OpenCV is used to read/write images from disk.  Execute `sudo apt install libopencv-dev`.
2. A cuda enabled GPU with a cuda toolkit installed. See the [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for additional details.  Basically any version
should work as the operations are very basic.

## Compiling
```angular2html
mkdir build && cd build
cmake ..
make && cd ..
```
## Usage

### help
```./barco -h```

### Applying tint to an image
```angular2html
./barco --source=./images/golden_retriever_puppy.png --destination=red_puppy.png --tint=255,0,0 --weight=0.3
```

Note, the `--tint=255,0,0` is applied in an RGB order and not a BGR ordering (OpenCV reads images in BGR order, this is accounted for in the program)

### Applying colour space conversion from BT.709 to BT.2020
```angular2html
 ./barco --source=./images/golden_retriever_puppy.png --destination=r2020_puppy.png --colour=true
```
### Samples
The [./images](./images) directory contains sample images to test on.

## Sources
- [RGB Color Space Conversion](https://www.ryanjuckett.com/rgb-color-space-conversion/)
- [Rec. 709](https://en.wikipedia.org/wiki/Rec._709)
- [Rec. 2020](https://en.wikipedia.org/wiki/Rec._2020)
