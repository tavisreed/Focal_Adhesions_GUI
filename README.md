# Focal_Adhesions_GUI
 
This is a github for the a trained neural network for the segmentation of focal adheisions. To use the neural network, follow the instructions below.

1. Download the entire GUI folder.
2. Create a python enviorment. I would suggest using anaconda to do this (https://anaconda.org/).
3. Download the necessary libraries into your enviorment. I don't have a complete list of libraries, but you can simply look at the code, or at the generated error messages to figure out which ones are needed. IMPORTANT NOTE: You must use a tensorflow version that is <2.0. Using any tensorflow version >=2.0 will likely cause errors, unless you edit the neural network code to be compatible, which is not suggested for casual users
4. Place the images you want to be segmented and tracked in the Raw_Images folder within their own named folder. Don't use spaces in the folder name (For example use "New_Images" instead of "New Images". The images themselves should have one channel, and be numbered numerically from first to last frame.
5. In the GUI, use the checkboxes to select what you want to have happen. For example, check "Segment Images" and "Track Segmented Images" to segement and track focal adhesions.
6. Whenever you are asked for a name in the GUI, be sure to use something unique, as the program will overwrite any existing files with the same name.
7. When you are asked for a folder path, if you only need to provide name of the folder you created. So for example, if you stored your raw images in "Foo", simply type "Foo" as the path. If you have multiple layers of folders, then provide that path. So to reach folder "Banana" within "Foo", type the path as "Foo\\Banana".
8. Running "Segment Images" will create a folder of segmented images in the "Masks" folder. Running "Track Segmented Images" will create folders "Tracked" containing images of tracked focal adhesions, and "CSVFiles" which contains information about all tracked objects in each frame.
9. To train a new model, provide both raw images and masks (hand segmented images) in the "Train" folder, following the same folder set up as the "Example Folder". The file numbers should correspond between the raw images and masks.
10. When training, the "Show Advanced Training Options" bar can be used to modify a select number of hyperparameters. The Image Split Size, is used to determine the size of the images that are acutally seen by the neural network. For example at value 64, a 128x128 image is processed as four 64x64 images. The threshhold values determine how confident the model must be about a pixel before calling it a focal adhesion. So for example when set yo 0.99, the model must be 99% certain that a pixel is part of a focal adhesion before assigning it a value of 1. The Minimum Percent Of White Pixels determine how much of an image must contain focal adhesions in order to be included as a training image. The rest of the parameters are standard deep learning parameters.

![Image of GUI](/gui_image.png)
