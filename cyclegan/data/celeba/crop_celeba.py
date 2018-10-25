import cv2
import os

# Get user supplied values
# Create the haar cascade
cascPath = '../util/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

dirname = 'CelebA/images'

if not os.path.exists('./images'):
    os.makedirs('./images')

# Read the image
for fn in sorted(os.listdir(dirname)):
    print(fn)
    image = cv2.imread(dirname + "/" + fn)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,5, 5)

    if len(faces) == 0:
        pass
    else:
        x, y, w, h = faces[0]
        image_crop = image[y:y+w, x:x+w, :]
        image_resize = cv2.resize(image_crop, (80, 80))
        fname = './images/' + fn[:-4] + '_crop' + fn[-4:]
        fname = fname.replace('.jpg', '.png')
        cv2.imwrite(fname, image_resize)



