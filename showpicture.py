from PIL import Image
import struct
#this file need publish ,so use my path
# add function1
path =my path

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

        # print('save ' + str(i) + 'image')
        image.save(save path)
        
        
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

    read_image(path)
    read_label(filepah1,filepath2)
