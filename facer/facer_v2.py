import cv2
import dlib
#import matplotlib.pyplot as plt
#from matplotlib import animation
import numpy as np
import math
import os
import glob

from facer.utils import similarityTransform, constrainPoint, calculateDelaunayTriangles, warpTriangle

# https://www.learnopencv.com/facial-landmark-detection/
# https://www.learnopencv.com/average-face-opencv-c-python-tutorial/
def load_face_detector():
    """Loads the dlib face detector"""
    return dlib.get_frontal_face_detector()

def load_landmark_predictor(predictor_path):
    return dlib.shape_predictor(predictor_path)

# Load the face detector and landmark predictor
PREDICTOR_PATH = "./module/shape_predictor_68_face_landmarks.dat"
detector = load_face_detector()
predictor = load_landmark_predictor(PREDICTOR_PATH)
print("Done, models loaded.")

def save_landmarks_to_disk(points, fp):
    txt = "\n".join(list(map(lambda p: f"{p.x}, {p.y}", (points))))
    with open(fp, "w") as outfile:
        outfile.write(txt)

def glob_image_files(root, extensions=["jpg", "jpeg", "png"]):
    """Returns a list of image files in `root`"""
    files = glob.glob(os.path.join(root, "*"))
    return [f for f in files if (f.rsplit(".", 1)[-1]).lower() in extensions]

def load_images(root, verbose=True):
    """Returns list of image arrays
    :param root: (str) Directory containing face images
    :param verbose: (bool) Toggle verbosity
    :output images: (dict) Dict of OpenCV image arrays, key is filename
    """
    print("find image with extensions (jpg, jpeg, png)")

    files = sorted(glob_image_files(root))
    num_files = len(files)
    if verbose:
        print(f"\nFound {num_files} in '{root}'.")
        N = max(round(0.10 * num_files), 1)

    # Load the images
    images = {}
    for n, file in enumerate(files):
        if verbose and n % N == 0:
            print(f"({n + 1} / {num_files}): {file}")
        image = cv2.imread(file)
        if image is None:
            continue
        if len(image.shape) != 3:
            continue

        h = image.shape[0]
        w = image.shape[1]
        max_side = max(h,w)
        scale = 1
        if max_side > 1024:
            scale = 1024 / max_side
        new_h = h * scale
        new_w = w * scale
        if scale != 1:
            image = cv2.resize(image, (int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
        image = image[..., ::-1]
        image = image / 255.0
        images[file] = image
    return images

def load_face_landmarks(root, verbose=True):
    """Load face landmarks created by `detect_face_landmarks()`
    :param root: (str) Path to folder containing CSV landmark files
    :param verbose: (bool) Toggle verbosity
    :output landmarks: (list)
    """
    #List all files in the directory and read points from text files one by one
    all_paths = glob.glob(root.strip("/") + "/*_landmarks*")
    print(all_paths)
    landmarks = []
    for fn in all_paths:
        points = []
        with open(fn) as file:
            for line in file:
                x, y = line.split(", ")
                points.append((int(x), int(y)))

        # Store array of points
        landmarks.append(points)
    return landmarks

def detect_face_landmarks(images,
                          save_landmarks=False,
                          max_faces=1,
                          verbose=True,
                          print_freq=0.10):
    """Detect and save the face landmarks for each image
    :param images: (dict) Dict of image files and arrays from `load_images()`.
    :param save_landmarks: (bool) Save landmarks to .CSV
    :param max_faces: (int) Skip images with too many faces found.
    :param verbose: (bool) Toggle verbosity
    :param print_freq: (float) How often do you want print statements?
    :output landmarks: (list) 68 landmarks for each found face
    :output faces: (list) List of the detected face images
    """
    num_images = len(images.keys())
    if verbose:
        print(f"\nStarting face landmark detection...")
        print(f"Processing {num_images} images.")
        N = max(round(print_freq * num_images), 1)

    # Look for face landmarks in each image
    num_skips = 0
    all_landmarks, all_faces = [], []
    for n, (file, image) in enumerate(images.items()):
        if verbose and n % N == 0:
            print(f"({n + 1} / {num_images}): {file}")

        # Try to detect a face in the image
        #imageForDlib = dlib.load_rgb_image(file) # Kludge for now
        imageForDlib = image * 255
        imageForDlib = imageForDlib.astype(np.uint8)
        #print()
        found_faces = detector(imageForDlib, 1)

        # Only save landmarks when num_faces = 1
        if len(found_faces) == 0 or len(found_faces) > max_faces:
            num_skips += 1
            continue

        # Find landmarks, save to CSV
        for num, face in enumerate(found_faces):
            landmarks = predictor(imageForDlib, face)
            if not landmarks:
                continue

            # Add this image to be averaged later
            all_faces.append(image)

            # Convert landmarks to list of (x, y) tuples
            lm = [(point.x, point.y) for point in landmarks.parts()]
            all_landmarks.append(lm)

            # Save landmarks as a CSV file (optional)
            if save_landmarks:
                fp = file.rsplit(".", 1)[0] + f"_landmarks_{num}.csv"
                save_landmarks_to_disk(landmarks.parts(), fp=fp)

    if verbose:
        print(f"Skipped {100 * (num_skips / num_images):.1f}% of images.")
    return all_landmarks, all_faces

def create_average_face(faces,
                        landmarks,
                        output_dims=(800, 800),
                        save_image=True,
                        output_file="average_face.jpg",
                        return_intermediates=False,
                        verbose=True,
                        print_freq=0.05):
    """Combine the faces into an average face"""
    if verbose:
        print(f"\nStarting face averaging for {len(faces)} faces.")
    msg = "Number of landmark sets != number of images."
    assert len(faces) == len(landmarks), msg

    # Eye corners
    num_images = len(faces)
    n = len(landmarks[0])
    w, h = output_dims
    eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)),
                    (np.int(0.7 * w), np.int(h / 3))]
    imagesNorm, pointsNorm = [], []

    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0,0),
                            (w / 2, 0),
                            (w - 1, 0),
                            (w - 1, h / 2),
                            (w - 1, h - 1),
                            (w / 2, h - 1),
                            (0, h - 1),
                            (0, h / 2)])

    # Initialize location of average points to 0s
    pointsAvg = np.array([(0, 0)] * (len(landmarks[0]) + len(boundaryPts)), np.float32())

    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    warped, incremental = [], []
    N = max(round(print_freq * num_images), 1)
    for i in range(0, num_images):
        if verbose and i % N == 0:
            print(f"Image {i + 1} / {num_images}")

        # Corners of the eye in input image
        points1 = landmarks[i]
        eyecornerSrc  = [landmarks[i][36], landmarks[i][45]]

        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)

        # Apply similarity transformation
        img_affine = cv2.warpAffine(faces[i], tform, (w, h)).get()

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68, 1, 2))
        points = cv2.transform(points2, tform).get()
        points = np.float32(np.reshape(points, (68, 2)))

        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundaryPts, axis=0)

        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / num_images
        pointsNorm.append(points)
        imagesNorm.append(img_affine)

        if return_intermediates:
            warped.append(img_affine)

    # Delaunay triangulation
    rect = (0, 0, w, h);
    #dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))
    dt = [
        (18,37,36),(37,18,19),(2,75,1),(75,2,3),(0,75,68),(75,0,1),(51,33,52),(33,51,50),(1,0,36),
        (2,31,3),(31,2,41),(75,3,4),(2,1,41),(67,62,66),(62,67,61),(74,4,5),(4,74,75),(49,61,67),
        (61,49,50),(74,5,6),(4,3,48),(35,13,54),(13,35,14),(73,74,7),(5,4,48),(41,1,36),(74,6,7),
        (6,5,59),(29,31,40),(31,29,30),(7,6,58),(37,40,41),(40,37,38),(73,7,8),(8,7,57),(10,73,9),
        (73,10,72),(9,73,8),(9,8,56),(11,72,10),(72,11,12),(10,9,56),(31,41,40),(11,10,55),(39,27,28),
        (27,39,21),(72,12,71),(12,11,54),(29,40,39),(71,12,13),(13,12,54),(18,36,17),(70,71,16),(13,14,71),
        (17,36,0),(15,71,14),(71,15,16),(15,14,46),(16,15,45),(22,69,23),(69,22,21),(0,68,17),(46,14,35),
        (18,69,19),(69,18,68),(42,29,28),(29,42,47),(18,17,68),(23,69,24),(19,69,20),(19,20,37),(21,22,27),
        (42,22,43),(22,42,27),(20,21,38),(21,20,69),(27,42,28),(29,47,35),(23,43,22),(43,23,44),(47,44,46),
        (44,47,43),(44,25,45),(25,44,24),(23,24,44),(24,69,25),(26,45,25),(45,26,16),(25,70,26),(70,25,69),
        (16,26,70),(35,30,29),(30,35,34),(11,55,54),(28,29,39),(34,52,33),(52,34,35),(59,5,48),(57,7,58),
        (31,50,49),(50,31,32),(31,30,32),(48,3,31),(32,30,33),(55,65,53),(65,55,56),(32,33,50),(33,30,34),
        (35,54,53),(40,38,39),(20,38,37),(36,37,41),(21,39,38),(46,44,45),(42,43,47),(15,46,45),(46,35,47),
        (48,31,49),(48,49,60),(62,61,51),(56,8,57),(50,51,61),(53,63,52),(63,53,65),(51,52,63),(52,35,53),
        (54,55,64),(53,54,64),(55,53,64),(55,10,56),(51,63,62),(49,67,59),(56,57,66),(49,59,60),(57,58,66),
        (58,6,59),(59,48,60),(58,59,67),(66,65,56),(65,66,62),(62,63,65),(66,58,67),
    ]
    # tmp_show = np.ones((h,w,3)) * 255
    # tmp_show = tmp_show.astype(np.uint8)
    # for i in range(len(dt)):
    #     d1 = tuple(map(lambda  x: int(x), pointsAvg[dt[i][0]]))
    #     d2 = tuple(map(lambda x: int(x), pointsAvg[dt[i][1]]))
    #     d3 = tuple(map(lambda x: int(x), pointsAvg[dt[i][2]]))
    #     cv2.line(tmp_show, d1, d2,1,4)
    #     cv2.line(tmp_show, d2, d3, 1, 4)
    #     cv2.line(tmp_show, d1, d3, 1, 4)
    # cv2.imwrite("debug_triangle.jpg", tmp_show)
    # return

    # Warp input images to average image landmarks
    output = np.zeros((h, w, 3), np.float32())
    for i in range(0, len(imagesNorm)):
        img = np.zeros((h, w, 3), np.float32())
        # Transform triangles one by one
        for j in range(0, len(dt)):
            tin, tout = [], []

            for k in range(0, 3):
                pIn = pointsNorm[i][dt[j][k]]
                pIn = constrainPoint(pIn, w, h)

                pOut = pointsAvg[dt[j][k]]
                pOut = constrainPoint(pOut, w, h)

                tin.append(pIn)
                tout.append(pOut)
            img = warpTriangle(imagesNorm[i], img, tin, tout)
        if return_intermediates:
            incremental.append((output + img) / (i + 1))


        # Add image intensities for averaging
        output = output + img

    # Divide by num_images to get average
    output = output / num_images

    if return_intermediates:
        incremental = incremental[-num_images:]
    print('Done.')

    # Save the output image to disk
    if save_image:
        cv2.imwrite(output_file, 255 * output[..., ::-1])
    if return_intermediates: # For animated GIFs
        return output, warped, incremental, imagesNorm
    return output

def create_average_face_from_directory(dir_in,
                                       dir_out,
                                       filename,
                                       save_image=True,
                                       **kwargs):
    verbose = kwargs.get('verbose', True)
    if verbose:
        print(f"Directory: {dir_in}")
    images = load_images(dir_in, verbose=verbose)
    if len(images) == 0:
        if verbose:
            print(f"Couldn't find any images in: '{dir_in}'.")
        return

    # Detect landmarks for each face
    landmarks, faces = detect_face_landmarks(images, verbose=verbose)

    if(len(faces) == 0):
        print("Not Found Face Images, Return")
        return None

    # Use  the detected landmarks to create an average face
    fn = f"average_face_{filename}.jpg"
    fp = os.path.join(dir_out, fn).replace(" ", "_")
    average_face = create_average_face(faces,
                                       landmarks,
                                       output_file=fp,
                                       save_image=True)

    # Save a labeled version of the average face
    #if save_image:
    #    save_labeled_face_image(average_face, filename, dir_out)
    return average_face


'''////////////////////////////////////////////////////////////////////////////////////////'''

def resize_image(image, isNeedNormalize=False):
    h = image.shape[0]
    w = image.shape[1]
    max_side = max(h, w)
    scale = 1
    if max_side > 1024:
        scale = 1024 / max_side
    new_h = h * scale
    new_w = w * scale
    if scale != 1:
        image = cv2.resize(image, (int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
    image = image[..., ::-1]
    if isNeedNormalize:
        image = image / 255.0
    return image

def single_image_detect_landmarks(image,
                          save_landmarks=False,
                          max_faces=1,
                          verbose=True):
    """Detect and save the face landmarks for each image
    :param images: (dict) Dict of image files and arrays from `load_images()`.
    :param save_landmarks: (bool) Save landmarks to .CSV
    :param max_faces: (int) Skip images with too many faces found.
    :param verbose: (bool) Toggle verbosity
    :param print_freq: (float) How often do you want print statements?
    :output landmarks: (list) 68 landmarks for each found face
    :output faces: (list) List of the detected face images
    """

    # face landmarks for image
    all_landmarks = []
    image = resize_image(image, isNeedNormalize=False)
    found_faces = detector(image, 1)
    if len(found_faces) == 0 or len(found_faces) > max_faces:
        return None

    # Find landmarks, save to CSV
    for num, face in enumerate(found_faces):
        landmarks = predictor(image, face)
        if not landmarks:
            continue
        # Convert landmarks to list of (x, y) tuples
        lm = [(point.x, point.y) for point in landmarks.parts()]
        all_landmarks.append(lm)
        # Save landmarks as a CSV file (optional)
        # if save_landmarks:
        #     fp = file.rsplit(".", 1)[0] + f"_landmarks_{num}.csv"
        #     save_landmarks_to_disk(landmarks.parts(), fp=fp)

    if len(all_landmarks) == 0:
        return None
    else:
        return all_landmarks


def load_images_and_landmark_detector(root, verbose=True):
    """Returns list of image arrays
    :param root: (str) Directory containing face images
    :param verbose: (bool) Toggle verbosity
    :output images: (dict) Dict of OpenCV image arrays, key is filename
    """
    print("find image with extensions (jpg, jpeg, png)")

    files = sorted(glob_image_files(root))
    num_files = len(files)
    if verbose:
        print(f"\nFound {num_files} in '{root}' for face detect and landmark")
        N = max(round(0.10 * num_files), 1)

    # Load the images
    image_dict = {}
    image_files = []
    landmarks = []
    for n, file in enumerate(files):
        if verbose and n % N == 0:
            print(f"({n + 1} / {num_files}): {file}")
        image = cv2.imread(file)
        if image is None:
            continue
        if len(image.shape) != 3:
            continue

        img_lands = single_image_detect_landmarks(image,save_landmarks=False,max_faces=1,verbose=True)
        if img_lands is not None:
            for i in range(len(img_lands)):
                image_files.append(file)
                landmarks.append(img_lands[i])
    image_dict['files'] = image_files
    image_dict['landmarks'] = landmarks
    return image_dict

def create_average_face_v2(image_dict,
                        output_dims=(800, 800),
                        save_image=True,
                        output_file="average_face.jpg",
                        return_intermediates=False,
                        verbose=True,
                        print_freq=0.05):
    """Combine the faces into an average face"""
    if verbose:
        print(f"\nStarting face averaging for {len(image_dict['files'])} faces.")
    msg = "Number of landmark sets != number of images."
    assert len(image_dict['files']) == len(image_dict['landmarks']), msg

    files_all = image_dict['files']
    landmarks_all = image_dict['landmarks']

    # Eye corners
    num_images = len(files_all)
    n = len(landmarks_all[0])
    w, h = output_dims
    eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)),
                    (np.int(0.7 * w), np.int(h / 3))]
    imagesNorm, pointsNorm = [], []

    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0,0),
                            (w / 2, 0),
                            (w - 1, 0),
                            (w - 1, h / 2),
                            (w - 1, h - 1),
                            (w / 2, h - 1),
                            (0, h - 1),
                            (0, h / 2)])

    # Initialize location of average points to 0s
    pointsAvg = np.array([(0, 0)] * (len(landmarks_all[0]) + len(boundaryPts)), np.float32())

    N = max(round(print_freq * num_images), 1)

    transforms = []
    if verbose:
        print(f"\n1) Average Point Calculate.")
    for i in range(0, num_images):
        if verbose and i % N == 0:
            print(f"Image {i + 1} / {num_images}")
        # Corners of the eye in input image
        points1 = landmarks_all[i]
        eyecornerSrc  = [landmarks_all[i][36], landmarks_all[i][45]]

        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)
        transforms.append(tform)

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68, 1, 2))
        points = cv2.transform(points2, tform).get()
        points = np.float32(np.reshape(points, (68, 2)))

        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundaryPts, axis=0)

        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / num_images
        pointsNorm.append(points)

    # Delaunay triangulation
    rect = (0, 0, w, h)
    #dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))
    dt = [
        (18,37,36),(37,18,19),(2,75,1),(75,2,3),(0,75,68),(75,0,1),(51,33,52),(33,51,50),(1,0,36),
        (2,31,3),(31,2,41),(75,3,4),(2,1,41),(67,62,66),(62,67,61),(74,4,5),(4,74,75),(49,61,67),
        (61,49,50),(74,5,6),(4,3,48),(35,13,54),(13,35,14),(73,74,7),(5,4,48),(41,1,36),(74,6,7),
        (6,5,59),(29,31,40),(31,29,30),(7,6,58),(37,40,41),(40,37,38),(73,7,8),(8,7,57),(10,73,9),
        (73,10,72),(9,73,8),(9,8,56),(11,72,10),(72,11,12),(10,9,56),(31,41,40),(11,10,55),(39,27,28),
        (27,39,21),(72,12,71),(12,11,54),(29,40,39),(71,12,13),(13,12,54),(18,36,17),(70,71,16),(13,14,71),
        (17,36,0),(15,71,14),(71,15,16),(15,14,46),(16,15,45),(22,69,23),(69,22,21),(0,68,17),(46,14,35),
        (18,69,19),(69,18,68),(42,29,28),(29,42,47),(18,17,68),(23,69,24),(19,69,20),(19,20,37),(21,22,27),
        (42,22,43),(22,42,27),(20,21,38),(21,20,69),(27,42,28),(29,47,35),(23,43,22),(43,23,44),(47,44,46),
        (44,47,43),(44,25,45),(25,44,24),(23,24,44),(24,69,25),(26,45,25),(45,26,16),(25,70,26),(70,25,69),
        (16,26,70),(35,30,29),(30,35,34),(11,55,54),(28,29,39),(34,52,33),(52,34,35),(59,5,48),(57,7,58),
        (31,50,49),(50,31,32),(31,30,32),(48,3,31),(32,30,33),(55,65,53),(65,55,56),(32,33,50),(33,30,34),
        (35,54,53),(40,38,39),(20,38,37),(36,37,41),(21,39,38),(46,44,45),(42,43,47),(15,46,45),(46,35,47),
        (48,31,49),(48,49,60),(62,61,51),(56,8,57),(50,51,61),(53,63,52),(63,53,65),(51,52,63),(52,35,53),
        (54,55,64),(53,54,64),(55,53,64),(55,10,56),(51,63,62),(49,67,59),(56,57,66),(49,59,60),(57,58,66),
        (58,6,59),(59,48,60),(58,59,67),(66,65,56),(65,66,62),(62,63,65),(66,58,67),
    ]

    # Warp input images to average image landmarks
    output = np.zeros((h, w, 3), np.float32())
    if verbose:
        print(f"\n2)Face Wrap and Averaging.")
    for i in range(0, num_images):
        if verbose and i % N == 0:
            print(f"Image {i + 1} / {num_images}")

        file = files_all[i]
        image = cv2.imread(file)

        image = resize_image(image, isNeedNormalize=True)
        tform = transforms[i]

        # Apply similarity transformation
        img_affine = cv2.warpAffine(image, tform, (w, h)).get()

        img = np.zeros((h, w, 3), np.float32())
        # Transform triangles one by one
        for j in range(0, len(dt)):
            tin, tout = [], []
            for k in range(0, 3):
                pIn = pointsNorm[i][dt[j][k]]
                pIn = constrainPoint(pIn, w, h)

                pOut = pointsAvg[dt[j][k]]
                pOut = constrainPoint(pOut, w, h)

                tin.append(pIn)
                tout.append(pOut)
            img = warpTriangle(img_affine, img, tin, tout)
        # Add image intensities for averaging
        output = output + img

    # Divide by num_images to get average
    output = output / num_images
    print('Done.')

    # Save the output image to disk
    if save_image:
        output = 255 * output[..., ::-1]
        output = np.clip(output, 0, 255).astype(np.uint8)
        cv2.imwrite(output_file, output)
    return output

def create_average_face_from_directory_v2(dir_in,
                                       dir_out,
                                       filename,
                                       save_image=True,
                                       **kwargs):
    verbose = kwargs.get('verbose', True)
    if verbose:
        print(f"Directory: {dir_in}")

    img_dict = load_images_and_landmark_detector(dir_in, verbose=True)
    if len(img_dict['files']) == 0:
        if verbose:
            print("Not Found Face Images, Return")
        return

    # Use  the detected landmarks to create an average face
    fn = f"average_face_{filename}.jpg"
    fp = os.path.join(dir_out, fn).replace(" ", "_")
    average_face = create_average_face_v2(img_dict,
                                       output_file=fp,
                                       save_image=True)

    return average_face
