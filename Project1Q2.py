import cv2
import numpy as np
import sys
import math

# histogram = {}
# for i in range(0, 100):
#     histogram[i] = 0


# the max and min lum value within H1, H2, W1, W2 windows, and the histogram
def max_min_histogram(H1, H2, W1, W2, image):
    min = 256
    max = -1
    histogram = {}
    for i in range(0, 101):
        histogram[i] = 0
    # histogram[round(100.0)] = 2
    for i in range(H1, H2):
        for j in range(W1, W2):
            l, u, v = image[i, j]
            histogram[round(l)] += 1
            if l >= max:
                max = l
            elif l <= min:
                min = l
    return max, min, histogram


def cumulative(max, min, histogram):
    cumulative = {}
    for i in range(0, 101):
        cumulative[i] = 0
    for i in range(min, max + 1):
        pre = 0
        if i > 0:
            pre = cumulative[i - 1]
        cumulative[i] = histogram[i] + pre
    return cumulative

def equalization(max, min, cumulative, n):
    new_value = {}
    for i in range(0, 101):
        new_value[i] = 0
    for i in range(min, max + 1):
        pre = 0
        if i > 0:
            pre = cumulative[i - 1]
        new_value[i] = math.floor(((pre + cumulative[i])/2) * (max - min + 1) / n)
    return new_value

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
n = (W2 - W1) * (H2 - H1)


tmp = np.copy(inputImage_BGR)
tmp = cv2.normalize(tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)
cv2.imshow("input", inputImage_BGR)


#Transform the color domain to LUV
inputImage_LUV = cv2.cvtColor(tmp, cv2.COLOR_BGR2LUV)

# get the local max and min within the windows, and the histogram
local_max, local_min, histogram = max_min_histogram(H1, H2, W1, W2, inputImage_LUV)
#round them and convert to int
local_max = int(round(local_max))
local_min = int(round(local_min))
#cumulative value for the histogram
cumulative = cumulative(local_max, local_min, histogram)

#new value mapping after equalization
new_value = equalization(local_max, local_min, cumulative, n)

newImage = np.zeros([rows, cols, bands], dtype=np.float32)
for i in range(0, rows):
    for j in range(0, cols):
        l, u, v = inputImage_LUV[i, j]
        if l < local_min:
            l = 0
        elif l > local_max:
            l = 100
        else:
            l = new_value[int(round(l))]
        newImage[i, j] = l, u, v
newImage = cv2.cvtColor(newImage, cv2.COLOR_LUV2BGR)
newImage = cv2.normalize(newImage, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U, dst=None)

cv2.imshow("output", newImage)
cv2.imwrite(name_output, newImage)



# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()