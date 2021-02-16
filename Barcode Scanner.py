import sys

from matplotlib import pyplot
from matplotlib.patches import Rectangle
# this is our module that performs the reading of a png image
import imageIO.png

#To change the image, change the image name in the main function.

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# This method takes a greyscale pixel array and writes it into a png file
def writeGreyscalePixelArraytoPNG(output_filename, pixel_array, image_width, image_height):
    file = open(output_filename, 'wb')  # binary mode is important
    writer = imageIO.png.Writer(image_width, image_height, greyscale=True)
    writer.write(file, pixel_array)
    file.close()


# Creates a two dimensional array representing an image as a very simple (not very efficient) list of lists
# datastructure.
# The outer list is covering all the image rows. Each row is an inner list covering the columns of the image.
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_array = []
    for row in range(image_height):
        new_row = []
        for col in range(image_width):
            new_row.append(initValue)
        new_array.append(new_row)

    return new_array


# Takes as input a greyscale pixel array and computes the minimum and maximum greyvalue.
# Returns minimum and maximum greyvalue as a tuple
def computeMinAndMaxValues(pixel_array, image_width, image_height):
    min_value = sys.maxsize
    max_value = -min_value

    for y in range(image_height):
        for x in range(image_width):
            if pixel_array[y][x] < min_value:
                min_value = pixel_array[y][x]
            if pixel_array[y][x] > max_value:
                max_value = pixel_array[y][x]

    return(min_value, max_value)


# This function analyzes the return value of the connected component label algorithm to derive the 
# bounding box around the largest connected component. Thus, it prepares the result to be shown 
# using a rectangle around the detected barcode.
def determineLargestConnectedComponent(cclabeled, label_size_dictionary, image_width, image_height):

    final_labeled = createInitializedGreyscalePixelArray(image_width, image_height)

    size_of_largest_component = 0
    label_of_largest_component = 0
    for lbl_i in label_size_dictionary.keys():
        if label_size_dictionary[lbl_i] > size_of_largest_component:
            size_of_largest_component = label_size_dictionary[lbl_i]
            label_of_largest_component = lbl_i

    print("label of largest component: ", label_of_largest_component)

    # determine bounding box of the largest component only
    bbox_min_x = image_width
    bbox_min_y = image_height
    bbox_max_x = 0
    bbox_max_y = 0
    for y in range(image_height):
        for x in range(image_width):
            if cclabeled[y][x] == label_of_largest_component:
                final_labeled[y][x] = 255
                if x < bbox_min_x:
                    bbox_min_x = x
                if y < bbox_min_y:
                    bbox_min_y = y
                if x > bbox_max_x:
                    bbox_max_x = x
                if y > bbox_max_y:
                    bbox_max_y = y
            else:
                final_labeled[y][x] = 0
    return (final_labeled, (bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y))


# a simple Queue datastructure based on a list, not very efficient but sufficient
# for a simple connected component labeling implementation
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
        
#Computes an RBG set of arrays to Greyscale and returns a greyscale pixel array
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):

    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    for y in range(image_height):
        for x in range(image_width):
            r = pixel_array_r[y][x]
            g = pixel_array_g[y][x]
            b = pixel_array_b[y][x]
            g = int(round(0.299*r+0.587*g+0.114*b))
            greyscale_pixel_array[y][x] = g

    return greyscale_pixel_array

#Min-max scaled to the full 8-bits(0-255). Returns the scaled 0-255 array
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):

    output_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)

    minMax = computeMinAndMaxValues(pixel_array, image_width, image_height)
    f_Low = minMax[0]
    f_High = minMax[1]
    gMin = 0
    gMax = 255
    for i in range(image_height):
        for j in range(image_width):
            try:
                sOut = round((pixel_array[i][j] - f_Low) * ((gMax - gMin) / (f_Low - f_High)) + gMin )
                output_pixel_array[i][j] = abs(sOut)
            except:
                output_pixel_array[i][j] = 0
    return output_pixel_array

# computes vertical edges using Sobel filter
# we ignore border pixels
def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):

    vertical_edges = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            if y==0 or y == image_height-1 or x == 0 or x == image_width-1:
                vertical_edges[y][x]=0
            else:
                vertical_edges[y][x] = abs((pixel_array[y-1][x-1]*1 + pixel_array[y-1][x]*0 + pixel_array[y-1][x+1]*-1 +
                pixel_array[y][x-1]*2 + pixel_array[y][x]*0 + pixel_array[y][x+1]*-2 +
                pixel_array[y+1][x-1]*1 +pixel_array[y+1][x]*0 + pixel_array[y+1][x+1]*-1)/8)
    return vertical_edges

# computes horizontal edges using Sobel filter
# we ignore border pixels
def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):

    horizontal_edges = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            if y==0 or y == image_height-1 or x == 0 or x == image_width-1:
                horizontal_edges[y][x]=0
                
            else:
                horizontal_edges[y][x] = abs((pixel_array[y-1][x-1]*1 + pixel_array[y-1][x]*2 + pixel_array[y-1][x+1]*1 +
                pixel_array[y][x-1]*0 + pixel_array[y][x]*0 + pixel_array[y][x+1]*0 +
                pixel_array[y+1][x-1]*-1 +pixel_array[y+1][x]*-2 + pixel_array[y+1][x+1]*-1)/8)
    return horizontal_edges


# takes vertical and horizontal edges as input and subtracts horizontal from vertical edges
# additionally, if this subtraction is negative, the value is set to 0
# assumes that vertical and horizontal edges are normalized!
# returns the subtracted image
def computeStrongVerticalEdgesBySubtractingHorizontal(vertical_edges, horizontal_edges, image_width, image_height):

    edges = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            val = vertical_edges[y][x] - horizontal_edges[y][x]
            if val < 0:
                edges[y][x] = 0
            else:
                edges[y][x] = val
    return edges

#Computes the box average and returns
def computeBoxAveraging3x3(pixel_array, image_width, image_height):

    averaged = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            if y==0 or y == image_height-1 or x == 0 or x == image_width-1:
                averaged[y][x]=0
                
            else:
                averaged[y][x] = abs((pixel_array[y-1][x-1]*1 + pixel_array[y-1][x]*1 + pixel_array[y-1][x+1]*1 +
                pixel_array[y][x-1]*1 + pixel_array[y][x]*1 + pixel_array[y][x+1]*1 +
                pixel_array[y+1][x-1]*1 +pixel_array[y+1][x]*1 + pixel_array[y+1][x+1]*1)/9)
    return averaged


# returns 255 for pixels greater or equal (GE) threshold value, 0 otherwise (strictly lower)
def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):

    thresholded = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(image_height):
        for x in range(image_width):
            
            if pixel_array[y][x] < threshold_value:
                thresholded[y][x]=0
            elif pixel_array[y][x] >=threshold_value:
                thresholded[y][x] =255
    
    return thresholded


def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):

    eroded = createInitializedGreyscalePixelArray(image_width, image_height)
    for y in range(1,image_height-1):
        for x in range(1,image_width-1):
            check = (pixel_array[y-1][x-1]*1 * pixel_array[y-1][x]*1 * pixel_array[y-1][x+1]*1 *
            pixel_array[y][x-1]*1 * pixel_array[y][x]*1 * pixel_array[y][x+1]*1 *
            pixel_array[y+1][x-1]*1 * pixel_array[y+1][x]*1 * pixel_array[y+1][x+1]*1)
            
            if check != 0:
                eroded[y][x] = 1
            else:
                eroded[y][x] = 0

    return eroded


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    
    dilated = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if i == image_height - 1 and j == image_width - 1:
                row1 = pixel_array[i-1][j-1] + pixel_array[i-1][j] + pixel_array[i-1][0]  
                row2 = pixel_array[i][j-1] + pixel_array[i][j] + pixel_array[i][0]
                row3 = pixel_array[0][j-1] + pixel_array[0][j] + pixel_array[0][0]
                kernel = row1 + row2 + row3
            elif i == image_height - 1:
                row1 = pixel_array[i-1][j-1] + pixel_array[i-1][j] + pixel_array[i-1][j+1]  
                row2 = pixel_array[i][j-1] + pixel_array[i][j] + pixel_array[i][j+1]
                row3 = pixel_array[0][j-1] + pixel_array[0][j] + pixel_array[0][j+1]
                kernel = row1 + row2 + row3
            elif j == image_width - 1:
                row1 = pixel_array[i-1][j-1] + pixel_array[i-1][j] + pixel_array[i-1][0]  
                row2 = pixel_array[i][j-1] + pixel_array[i][j] + pixel_array[i][0]
                row3 = pixel_array[i+1][j-1] + pixel_array[i+1][j] + pixel_array[i+1][0]
                kernel = row1 + row2 + row3
            else:   
                row1 = pixel_array[i-1][j-1] + pixel_array[i-1][j] + pixel_array[i-1][j+1]  
                row2 = pixel_array[i][j-1] + pixel_array[i][j] + pixel_array[i][j+1]  
                row3 = pixel_array[i+1][j-1] + pixel_array[i+1][j] + pixel_array[i+1][j+1]
                kernel = row1 + row2 + row3
            if kernel > 0:
                dilated[i][j] = 1

    return dilated


def computeConnectedComponentLabeling(binary_array, image_width, image_height):
    
    visitedArray = createInitializedGreyscalePixelArray(image_width, image_height)
    currentLabel = 1
    ccSizeDict = {}
    for y in range(image_height):
        for x in range(image_width):
            if visitedArray[y][x] == 0 and binary_array[y][x] != 0:
                ccSizeDict[currentLabel] = 0
                q = Queue()
                q.enqueue([y, x])
                binary_array[y][x] = currentLabel
                visitedArray[y][x] = 1
                while q.isEmpty() == False:
                    value = q.dequeue()
                    ccSizeDict[currentLabel] += 1
                    py = value[0]
                    px = value[1] 
                    if px-1 < 0:
                        pass
                    elif binary_array[py][px-1] != 0 and visitedArray[py][px-1] != 1:
                        visitedArray[py][px-1] = 1
                        binary_array[py][px-1] = currentLabel
                        q.enqueue([py, px-1])
                    if px == image_width - 1:
                        pass
                    elif binary_array[py][px+1] != 0 and visitedArray[py][px+1] != 1:
                        visitedArray[py][px+1] = 1
                        binary_array[py][px+1] = currentLabel
                        q.enqueue([py, px+1])

                    if py-1 < 0:
                        pass
                    elif binary_array[py-1][px] != 0 and visitedArray[py-1][px] != 1:
                        visitedArray[py-1][px] = 1
                        binary_array[py-1][px] = currentLabel
                        q.enqueue([py-1, px])

                    if py == image_height - 1:
                        pass
                    elif binary_array[py+1][px] != 0 and visitedArray[py+1][px] != 1:
                        visitedArray[py+1][px] = 1
                        binary_array[py+1][px] = currentLabel
                        q.enqueue([py+1, px])
                        
                currentLabel += 1
    return (binary_array, ccSizeDict)

def main():

    filename = "./images/barcodeDetection/barcode_05.png"

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(filename)

    # first we have to convert the red, green and blue pixel arrays to a greyscale representation.
    # This is done using the formula: greyvalue = 0.299 * red + 0.587 * green + 0.114 * blue
    px_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

    # next we make sure that the input greyscale image is scaled across the full 8 bit range (0 and 255)
    px_array = scaleTo0And255AndQuantize(px_array, image_width, image_height)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(3, 2)
    axs1[0, 0].set_title('Input greyscale image')
    axs1[0, 0].imshow(px_array, cmap='gray')

    # now we compute the horizontal edges in the image and take its absolute values...
    horizontal_edges = computeHorizontalEdgesSobelAbsolute(px_array, image_width, image_height)

    # scale horizontal edges to the range 0 and 255
    horizontal_edges = scaleTo0And255AndQuantize(horizontal_edges, image_width, image_height)

    # as well as the vertical edges in the image, again taking its absolute values.
    # TODO: implement this edge enhancement function
    vertical_edges = computeVerticalEdgesSobelAbsolute(px_array, image_width, image_height)
    # scale vertical edges to the range 0 and 255
    vertical_edges = scaleTo0And255AndQuantize(vertical_edges, image_width, image_height)

    # now we want to enhance strong vertical edges (our barcodes) by subtracting all horizontal edges
    edges = computeStrongVerticalEdgesBySubtractingHorizontal(vertical_edges, horizontal_edges, image_width, image_height)
    edges = scaleTo0And255AndQuantize(edges, image_width, image_height)

    # next we blur our edge image using a 3x3 mean filter (averaging or box filter) a total of four times
    # the result of the 3x3 mean filter ignores the border pixels, therefore the output is 0 along the image border
    averaged_edges = edges
    for i in range(10):
        averaged_edges = computeBoxAveraging3x3(averaged_edges, image_width, image_height)
    averaged_edges = scaleTo0And255AndQuantize(averaged_edges, image_width, image_height)

    axs1[0, 1].set_title('Averaged edge image')
    axs1[0, 1].imshow(averaged_edges, cmap='gray')

    # we use a threshold value of 70 to binarize the edge image. Note that this threshold depends crucially
    # on the fact that we are always working with normalized 8 bit images between 0 and 255
    threshold_value = 70
    thresholded = computeThresholdGE(averaged_edges, threshold_value, image_width, image_height)

    axs1[1, 0].set_title('Thresholded image')
    axs1[1, 0].imshow(thresholded, cmap='gray')

    eroded = thresholded
    for i in range(4):
        eroded = computeErosion8Nbh3x3FlatSE(eroded, image_width, image_height)
    dilated = eroded
    for i in range(4):
        dilated = computeDilation8Nbh3x3FlatSE(dilated, image_width, image_height)

    axs1[1, 1].set_title('Morphologically processed image')
    axs1[1, 1].imshow(dilated, cmap='gray')


    # taking the morphologically cleaned up binary image, we finally look for the largest connected component
    # in the image
    (cclabeled, size_dict_cc) = computeConnectedComponentLabeling(dilated, image_width, image_height)


    # inspect the result of the connected component labeling, derive the largest component and its bounding box
    (final_labeled, (bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y)) = \
        determineLargestConnectedComponent(cclabeled, size_dict_cc, image_width, image_height)

    axs1[2, 0].set_title('Largest detected component')
    axs1[2, 0].imshow(final_labeled, cmap='gray')

    print("bbox {} {} {} {}".format(bbox_min_x, bbox_max_x, bbox_min_y, bbox_max_y))

    # Draw the bounding box as a rectangle into the original input image
    axs1[2, 1].set_title('Final image of detection')
    axs1[2, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=3,
                     edgecolor='g', facecolor='none')
    axs1[2, 1].add_patch(rect)

    # plot the current figure
    pyplot.show()

if __name__ == "__main__":
    main()
