from __future__ import print_function
from builtins import input
import os, sys, importlib
import shutil, time
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)


####################################
# Parameters
####################################
boSaveDebugImg = True
subDirs = ['positive', 'testImages', 'negative']
image_sets = ["train", "test"]

# no need to change these parameters
boAddSelectiveSearchROIs = True
boAddRoisOnGrid = True


####################################
# Main
####################################
# generate ROIs using selective search and grid (for pascal we use the precomputed ROIs from Ross)
#if False: # for debugging
if not datasetName.startswith("pascalVoc"):
    # init
    makeDirectory(roiDir)
    roi_minDim = roi_minDimRel * roi_maxImgDim
    roi_maxDim = roi_maxDimRel * roi_maxImgDim
    roi_minNrPixels = roi_minNrPixelsRel * roi_maxImgDim*roi_maxImgDim
    roi_maxNrPixels = roi_maxNrPixelsRel * roi_maxImgDim*roi_maxImgDim

    for subdir in subDirs:
        makeDirectory(roiDir + subdir)
        imgFilenames = getFilesInDirectory(imgDir + subdir, ".jpg")

        # loop over all images
        for imgIndex,imgFilename in enumerate(imgFilenames):
            roiPath = "{}/{}/{}.roi.txt".format(roiDir, subdir, imgFilename[:-4])

            # load image
            print (imgIndex, len(imgFilenames), subdir, imgFilename)
            tstart = datetime.datetime.now()
            imgPath = imgDir + subdir + "/" + imgFilename
            imgOrig = imread(imgPath)
            if imWidth(imgPath) > imHeight(imgPath):
                print (imWidth(imgPath) , imHeight(imgPath))

            # get rois
            if boAddSelectiveSearchROIs:
                print ("Calling selective search..")
                rects, img, scale = getSelectiveSearchRois(imgOrig, ss_scale, ss_sigma, ss_minSize, roi_maxImgDim) #interpolation=cv2.INTER_AREA
                print ("   Number of rois detected using selective search: " + str(len(rects)))
            else:
                rects = []
                img, scale = imresizeMaxDim(imgOrig, roi_maxImgDim, boUpscale=True, interpolation=cv2.INTER_AREA)
            imgWidth, imgHeight = imArrayWidthHeight(img)

            # add grid rois
            if boAddRoisOnGrid:
                rectsGrid = getGridRois(imgWidth, imgHeight, grid_nrScales, grid_aspectRatios)
                print ("   Number of rois on grid added: " + str(len(rectsGrid)))
                rects += rectsGrid

            # run filter
            print ("   Number of rectangles before filtering  = " + str(len(rects)))
            rois = filterRois(rects, imgWidth, imgHeight, roi_minNrPixels, roi_maxNrPixels, roi_minDim, roi_maxDim, roi_maxAspectRatio)
            if len(rois) == 0: #make sure at least one roi returned per image
                rois = [[5, 5, imgWidth-5, imgHeight-5]]
            print ("   Number of rectangles after filtering  = " + str(len(rois)))

            # scale up to original size and save to disk
            # note: each rectangle is in original image format with [x,y,x2,y2]
            rois = np.int32(np.array(rois) / scale)
            assert (np.min(rois) >= 0)
            assert (np.max(rois[:, [0,2]]) < imArrayWidth(imgOrig))
            assert (np.max(rois[:, [1,3]]) < imArrayHeight(imgOrig))
            np.savetxt(roiPath, rois, fmt='%d')
            print ("   Time [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

# clear imdb cache and other files
if os.path.exists(cntkFilesDir):
    assert(cntkFilesDir.endswith("cntkFiles/"))
    userInput = input('--> INPUT: Press "y" to delete directory ' + cntkFilesDir + ": ")
    if userInput.lower() not in ['y', 'yes']:
        print ("User input is %s: exiting now." % userInput)
        exit(-1)
    shutil.rmtree(cntkFilesDir)
    time.sleep(0.1) # avoid access problems

# create cntk representation for each image
for image_set in image_sets:
    imdb = imdbs[image_set]
    print ("Number of images in set {} = {}".format(image_set, imdb.num_images))
    makeDirectory(cntkFilesDir)

    # open files for writing
    cntkImgsPath, cntkRoiCoordsPath, cntkRoiLabelsPath, cntkRegrTargetPath = getCntkInputPaths(cntkFilesDir, image_set)
    with open(cntkImgsPath, 'w')       as cntkImgsFile, \
         open(cntkRoiCoordsPath, 'w')  as cntkRoiCoordsFile, \
         open(cntkRoiLabelsPath, 'w')  as cntkRoiLabelsFile, \
         open(cntkRegrTargetPath, 'w') as regrTargetsFile:

        # for each image, transform rois etc to cntk format
        for imgIndex in range(0, imdb.num_images):
            if imgIndex % 50 == 0:
                print ("Processing image set '{}', image {} of {}".format(image_set, imgIndex, imdb.num_images))
            currBoxes = imdb.roidb[imgIndex]['boxes']
            currGtOverlaps = imdb.roidb[imgIndex]['gt_overlaps']
            gtm = imdb.roidb[imgIndex]['gt_argmaxes']
            imgPath = imdb.image_path_at(imgIndex)
            imgWidth, imgHeight = imWidthHeight(imgPath)

            # all rois need to be scaled + padded to cntk input image size
            w_offset, h_offset, scale = roiTransformPadScaleParams(imgWidth, imgHeight, cntk_padWidth, cntk_padHeight)

            num_rois = len(currBoxes)
            rel_coords = np.zeros((num_rois, 4), dtype=np.float32)
            for boxIndex, box in enumerate(currBoxes):
                coords = roiTransformPadScale(box, w_offset, h_offset, scale)
                rel_coords[boxIndex,:] = getCntkRelativeRoiCoords(coords, cntk_padWidth, cntk_padHeight)

            boxesStr = ""
            labelsStr = ""
            regrStr = ""
            for boxIndex, box in enumerate(currBoxes):
                roi_rel_coords = rel_coords[boxIndex]
                gt_rel_coords = rel_coords[gtm[boxIndex]]

                overlaps = currGtOverlaps[boxIndex, :].toarray()[0]
                label_wrt_overlap = getCntkRoiLabels(overlaps, train_posOverlapThres, nrClasses)

                regr_target = getBboxRegressionTarget(gt_rel_coords, roi_rel_coords)
                # if the candidate ROI is mapped to the background class the regression target is zero
                # [Note: background ROIs are ignored in the multi-task loss of Fast R-CNN]
                if label_wrt_overlap[0] == 1:
                    regr_target = (0.0, 0.0, 0.0, 0.0)
                # debug output
                #else:
                #    print(label_wrt_overlap)
                #    print(roi_rel_coords)
                #    print(gt_rel_coords)
                #    print(regr_target)
                #    print("---")

                boxesStr  += " {}".format(" ".join(str(x) for x in roi_rel_coords))
                labelsStr += " {}".format(" ".join(str(x) for x in label_wrt_overlap))
                regrStr   += " {}".format(" ".join(str(x) for x in regr_target))

            # if less than e.g. 2000 rois per image, then fill in the rest using 'zero-padding'.
            boxesStr, labelsStr, regrStr = cntkPadInputs(num_rois, cntk_nrRois, nrClasses, boxesStr, labelsStr, regrStr)

            # update cntk data
            cntkImgsFile.write("{}\t{}\t0\n".format(imgIndex, imgPath))
            cntkRoiCoordsFile.write("{} |rois{}\n".format(imgIndex, boxesStr))
            cntkRoiLabelsFile.write("{} |roiLabels{}\n".format(imgIndex, labelsStr))
            regrTargetsFile.write("{} |regrTarget{}\n".format(imgIndex, regrStr))

print ("DONE.")
