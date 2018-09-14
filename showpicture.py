from PIL import Image
import struct
#this file need publish ,so use my path
# add function1
path1 =r'D:\_____\Works\PyProject\tensortest\image_cv\ori_data'
path2='D:\_____\Works\PyProject\tensortest\image_cv\data'
def read_image(filename):
    f = open(filename, 'rb')

    index = 0
    buf = f.read()

    f.close()

    magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    for i in range(images):
        # for i in xrange(2000):
        image = Image.new('L', (columns, rows))

        for x in range(rows):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')

        #save train or test
        #image.save(path2+r'\train'+'\\'+ str(i) + '.png')
        image.save(path2 + r'\test' + '\\' + str(i) + '.png')
        
def read_label(filename, saveFilename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()

    f.close()

    magic, labels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')

    labelArr = [0] * labels
    # labelArr = [0] * 2000


    for x in range(labels):
        # for x in xrange(2000):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')

    save = open(saveFilename, 'w')

    save.write(','.join(map(lambda x: str(x), labelArr)))
    save.write('\n')

    save.close()
    print('save labels success')


if __name__ == '__main__':
    read_image(path1 + r'\t10k-images.idx3-ubyte')
    read_label(path1 + r'\t10k-labels.idx1-ubyte', path2 + r'\train_label.txt')
    read_image(path1 + r'\train-images.idx3-ubyte')
    read_label(path1 + r'\train-labels.idx1-ubyte', path2 + r'\train_label.txt')