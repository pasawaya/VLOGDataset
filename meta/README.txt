 _    ____    ____  ______
| |  / / /   / __ \/ ____/
| | / / /   / / / / / __  
| |/ / /___/ /_/ / /_/ /  
|___/_____/\____/\____/   
                        
-------------------------------------------------------------------------------
David F. Fouhey, Wei-cheng Kuo, Alexei A. Efros, Jitendra Malik
From Lifestyle VLOGs to Everyday Interaction
By using this data, you agree to be bound by the VLOG agreement.
Contact: dfouhey@eecs.berkeley.edu 
-------------------------------------------------------------------------------

0. About

This package contains version 1.1 of the metadata, labels, and data links for
the VLOG dataset. We may release additional labels and corrections to labels
(e.g., increased label precision on test samples) based on community interest.

This package contains urls to the data and frame ids. You must download
that data separately.

All VLOG data is stored in a format as follows. Suppose it is the Yth 
clip of a video with id ID, then the file-path is:

ID(-3)/ID(-2)/ID(-1)/ID/Y/

We provide a manifest of all the vidoes in the dataset, and other clip-level
metadata (e.g. labels, split information, etc.) are stored in the same order.
So, if X/I/4/v_mozQXWErXI4/016/ is the 200th line of manifest.txt (indexing by
1), then splitId[199] (indexing by 0) is the split id for that video, Y[199] is
its label, etc.

Version history:
    v1.1 (Mar 2018): 
        Full annotations for proxemics, scene categories
        Misc manifests for train/val/test
        Precomputed results
        Moved demo to github to separate code/data

    v1.0 (Dec 2017): 
        initial release

1. Files

youtube_links.txt -
    For every video, the youtube webpage url as well as frame start and frame
    end (inclusive). We downloaded and processed using unmodified youtube-dl
    and ffmpeg.

manifest.txt -  
    The manifest, giving the paths to all the videos. The location in this file
    serves as a key for all the other data.

manifest(Train/Val/Trainval/Test).txt -
    The manifest for the train, val, trainval, and test splits. 

splitId.txt - 
    The 4-way split for each video. 
        0 is test
        1,2,3 are trainval to be divided as you see fit

uploaderId.txt -
    A unique uploader id. This is done by taking the unique uploader names
    returned by youtube-dl. This may be of interest for more ensuring that you
    train and test on different people.

2. Label folders

We also provide folders containing a number of different tasks.

hand_object -
    Labels for the hand/object contact task. This contains hand_object.npy, 
    which is a Nx30 matrix L where the labeling indicates:

                {   1: video i contains hand contact with object j;
        L_ij =  {   0: labelers were inconclusive. Note that this occurs in 
                {      blocks since the labels were collected in blocks
                {  -1: video i does not contain hand contact with object j;

    It also contains hand_object_labels.txt, which gives the names of the 30 
    classes.

scene_category -
    Labels for ~20K clips in terms of scene category. This contains 
    scene_category_full.npy, a Nx1 matrix L where the labeling indicates:
        
        L_i = { k >= 0: this video is labeled as being the class k
              { k = -1: this video was not labeled or was inconclusive

    It also contains category_label.txt, which gives the names of the 6 classes

    scene_category.npy is the *old* 20K-only set that was annotated at the time
    of the initial collection.

scene_proxemic -
    Labels for in terms of scene proxemics. This contains
    scene_proxemic_full.npy, a Nx1 matrix L where the labeling indicates:
        
        L_i = { k >= 0: this video is labeled as being the class k
              { k = -1: this video was not labeled or was inconclusive

    It also contains proxemic_label.txt, which gives the names of the 4 classes.

        Intimate: z < 45cm
        Personal: 45cm < z < 1.2m
        Social:   1.2m < z < 3.7m
        Public:   3.7m < z
    
    These were labeled according to the nearest surface in the middle 75% of
    the picture in one frame from the video. 

    scene_proxemic.npy is the *old* 20K-only set that was annotated at the time
    of the intial collection.

hand_state -
    Labels for the hand-state prediction task. This contains hand_state.txt,
    which is of the format:

        image_path phase_name label_id

    It also contains hand_state_label.txt, which gives the names of the 
    classes. Note that touching something here means touching the external
    scene. This therefore excludes touching one's hair for instance.

hand_detect - 
    Bounding boxes for ~5K images. This contains hand_box_labels.txt, which
    is of the format:
        image_path split_id x1 y1 x2 y2 x1 y1 x2 y2 ...
    for 0 or more bounding boxes

3.

precomputed_results -
    Precomputed results for hand_boject and hand_state tasks for a variety of
    methods
