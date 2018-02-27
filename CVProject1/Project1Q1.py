import cv2
import numpy as np
import sys

# the max and min lum value within H1, H2, W1, W2 windows
def max_min_Lum(H1, H2, W1, W2, image):
    min = 256
    max = -1
    for i in range(H1, H2):
        for j in range(W1, W2):
            l, u, v = image[i, j]
            if l >= max:
                max = l
            elif l <= min:
                min = l
    return max, min

# apply linear scaling to the image
# x -> (x - a)(B - A)/(b - a) + A
def linear_scaling(max, min, image):
    rows, cols, bands = image.shape
    newImage = np.zeros([rows, cols, bands], dtype=np.float32)

    # scale it from 0 to 255, calculate the increment for each min to max, and calculate the corresponding mapped value
    increment = 100 / (max - min)
    for i in range(0, rows):
        for j in range(0, cols):
            l, u, v = image[i, j]
            if l < min:
                l = 0
            elif l > max:
                l = 100
            else:
                l = (l - min) * increment
            newImage[i, j] = l, u, v
    newImage = cv2.cvtColor(newImage, cv2.COLOR_LUV2BGR)
    newImage = cv2.normalize(newImage, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U, dst=None)
    return newImage

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage_BGR = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage_BGR is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()



rows, cols, bands = inputImage_BGR.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

cv2.imshow("input", inputImage_BGR)

tmp = np.copy(inputImage_BGR)
tmp = cv2.normalize(tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)



#Transform the color domain to LUV
inputImage_LUV = cv2.cvtColor(tmp, cv2.COLOR_BGR2LUV)


local_max, local_min = max_min_Lum(H1, H2, W1, W2, inputImage_LUV)
new_image = linear_scaling(local_max, local_min, inputImage_LUV)

cv2.imshow("output", new_image)
cv2.imwrite(name_output, new_image)



# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()