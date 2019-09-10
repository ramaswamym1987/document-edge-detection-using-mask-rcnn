# Document Edge Detection using Mask RCNN

## Steps:
1. Create a directory "MaskRCNN_DocEdgeDet" and move into it.

2. Clone the Mask_RCNN from matterport by typing the command "git clone https://github.com/matterport/Mask_RCNN".

3. Install the necessary libraries from the requirements.txt inside the "Mask_RCNN" using "pip install -r requirements.txt".

4. Then jump into the directory ".../Mask_RCNN/samples" (i.e. samples), where "ballon" directory can be found. Make a copy of it and rename it as "doc".

5. Open the "doc" directory and replace the string "ballon" with "doc" in all the names of the files available inside it.

6. Create two additional directories namely "coco" & "dataset" inside the "doc" directory. Then create two directories namely "train" & "val" inside the directoy "dataset".
   So the directory structure looks as below.

```
   Mask_RCNN -----> samples -----> doc
                                     |
                                     -----coco -----> mask_rcnn_coco.h5
                                     |
                                     ----->dataset
                                                 |
                                                 ----->train (contains all the train images along with the via_region_data.json(annotated file))
                                                 |
                                                 ----->test  (contains all the test images along with the via_region_data.json(annotated file))
```

7. Place all the train and test/validation images inside the directories 
  ``` ".../Mask_RCNN/samples/doc/dataset/train" & ".../Mask_RCNN/samples/doc/dataset/test". ```

8. For annotation of images (both train & val) goto to the website "http://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.6.html".
   Reference : "http://www.robots.ox.ac.uk/~vgg/software/via/" ---> Check for online copy here.
   Check the link ""https://www.youtube.com/watch?v=uUYR6IEm5VM"
     Save the annotated results as .json files (via_region_data.json) inside the respective "train" & "test" directories of "dataset" directory.


9. Open the file "doc.py" inside the directoy ".../Mask_RCNN/samples/doc". Here change all the strings "ballon/Ballon" to "doc/Document". Then, comment the below piece of code.
```
   # Load weights
   print("Loading weights ", weights_path)
   if args.weights.lower() == "coco":
       # Exclude the last layers because they require a matching
       # number of classes
       model.load_weights(weights_path, by_name=True, exclude=[
           "mrcnn_class_logits", "mrcnn_bbox_fc",
           "mrcnn_bbox", "mrcnn_mask"])
   else:
       model.load_weights(weights_path, by_name=True)
```

10. To start the training, from ".../Mask_RCNN/samples/doc" run the below command
```
python doc.py train --dataset=/Mask_RCNN/samples/doc/dataset --weights=/Mask_RCNN/samples/doc/coco/mask_rcnn_coco.h5
```



