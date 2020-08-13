from flexx import flx
from Train.train_model import train
from Segment.segmentor import segment
from Track.track import track



class TuningFork(flx.PyComponent):
    def init(self):
        super().init()
        with flx.VBox():

            flx.Label(style='border:1px solid #777;', wrap=1,
                    text='Tuning Fork: Biomedical Image Processing Software')

            flx.Label(style='border:1px solid #777;', wrap=1,
                      text='DESCRIPTION')
            with flx.HSplit(style='border:1px solid #777;', flex=0):
                with flx.HBox(flex=0):
                    self.TrainBox = flx.CheckBox(flex=0, text='Train Model On New Images')
                with flx.HBox(flex=0):
                    self.SegBox=flx.CheckBox(flex=0, text='Segment Images')
                with flx.HBox(flex=0):
                    self.TrackBox=flx.CheckBox(flex=0, text='Track Segmented Images')
                with flx.HBox(flex=0):
                    self.AnalyzeBox=flx.CheckBox(flex=0, text='Train ')
            with flx.HSplit(flex=1):
                ###############################################################################
                # Training
                ###############################################################################
                with flx.VBox(style='border:1px solid #777;'):
                    flx.Label(wrap=1, text='Training')

                    flx.Label(wrap=1, text='What You Want The New Model To Be Called')
                    self.TrainName = flx.LineEdit(placeholder_text='Model Name',
                                                  text='Example')

                    flx.Label(wrap=1, text='Folder Name Containing Training Images and Masks Folders')
                    self.TrainPath = flx.LineEdit(placeholder_text='Folder Name',
                                                  text='Example')

                    flx.Label(wrap=1, text='Number Of Color Channles In Images')
                    self.NumChannles = flx.LineEdit(placeholder_text='Number Of Channles',
                                                    text='1')

                    flx.Label(wrap=1, text='Training Image Type')
                    self.ImgType = flx.LineEdit(placeholder_text='Image Type, Ex. tif',
                                                text='tif')

                    flx.Label(wrap=1, text='Training Mask Type')
                    self.MaskType = flx.LineEdit(placeholder_text='Image Type, Ex. tif',
                                                 text='tif')

                    self.but_train_settings = flx.Button(text='Show Advanced Training Options')
                    self.showingCheck = False
                    self.temp=flx.VBox(style='border:1px solid #777; visibility: hidden;',)
                    with self.temp:
                        flx.Label(wrap=1, text='Below are the advanced settings. Only change these values if'
                                               'you know what you are doing!')
                        flx.Label(wrap=1, text='Image Split Size (64,128,256)')
                        self.ImageWidthHeight = flx.LineEdit(text='64')
                        flx.Label(wrap=1, text='Minimum Percent Of White Pixels')
                        self.MinPercentWhite = flx.LineEdit(text='0.16')
                        flx.Label(wrap=1, text='Percent Of Images To Use For Validation')
                        self.ValidationSplit = flx.LineEdit(text='0.15')
                        flx.Label(wrap=1, text='Small Model Kernal Size')
                        self.SmallKernal = flx.LineEdit(text='3')
                        flx.Label(wrap=1, text='Small Model Batch Size')
                        self.SmallBatch = flx.LineEdit(text='16')
                        flx.Label(wrap=1, text='Small Model Number Of Epochs')
                        self.SmallEpochs= flx.LineEdit(text='50')
                        flx.Label(wrap=1, text='Small Model Thresh Percentage')
                        self.SmallThresh= flx.LineEdit(text='0.8')
                        flx.Label(wrap=1, text='Boundary Model Kernal Size')
                        self.BoundaryKernal = flx.LineEdit(text='5')
                        flx.Label(wrap=1, text='Boundary Model W0 Value')
                        self.BoundaryW0 = flx.LineEdit(text='1')
                        flx.Label(wrap=1, text='Boundary Model Sigma Value')
                        self.BoundarySigma = flx.LineEdit(text='14')
                        flx.Label(wrap=1, text='Boundary Model Batch Size')
                        self.BoundaryBatch = flx.LineEdit(text='32')
                        flx.Label(wrap=1, text='Boundary Model Number Of Epochs')
                        self.BoundaryEpochs = flx.LineEdit(text='200')
                        flx.Label(wrap=1, text='Boundary Model Thresh Percentage')
                        self.BoundaryThresh = flx.LineEdit(text='0.999999977')
                        flx.Label(wrap=1, text='Mixer Model Kernal Size')
                        self.MixerKernal = flx.LineEdit(text='9')
                        flx.Label(wrap=1, text='Mixer Model Batch Size')
                        self.MixerBatch = flx.LineEdit(text='8')
                        flx.Label(wrap=1, text='Mixer Model Number Of Epochs')
                        self.MixerEpochs = flx.LineEdit(text='100')
                ###############################################################################
                # Segmenting
                ###############################################################################
                with flx.VBox(style='border:1px solid #777;'):
                    flx.Label(text='Segmenting')

                    flx.Label(wrap=1, text='Model Name')
                    self.SegmentName = flx.LineEdit(placeholder_text='Model Name',
                                                    text='Default')

                    flx.Label(wrap=1, text='Path To Folder Of Images To Segment')
                    self.SegmentImages= flx.LineEdit(placeholder_text='Path To Images',
                                                     text='Example')

                    flx.Label(wrap=1, text='File Type Of Images To Segment')
                    self.SegmentImgType = flx.LineEdit(placeholder_text='File Type',
                                                       text='tif')

                    flx.Label(wrap=1, text='Path To Folder To Save Masks In')
                    self.SegmentMaskSave = flx.LineEdit(placeholder_text='Path To Images',
                                                        text='Example')

                    flx.Label(wrap=1, text='Mask Save Type')
                    self.SegmentMaskType = flx.LineEdit(placeholder_text='File Type',
                                                        text='png')

                    flx.Label(wrap=1, text='Confidence Threshold')
                    self.SegmentThresh = flx.LineEdit(placeholder_text='Confidence Threshold',
                                                      text='0.9999')
                ###############################################################################
                # Tracking
                ###############################################################################
                with flx.VBox(style='border:1px solid #777;'):
                    flx.Label(text='Tracking')

                    flx.Label(wrap=1, text='Raw Images Folder Name')
                    self.TrackerImages = flx.LineEdit(placeholder_text='Folder Name',
                                                      text='Example')

                    flx.Label(wrap=1, text='Raw Image Type')
                    self.TrackerImageType = flx.LineEdit(placeholder_text='File Type',
                                                         text='tif')

                    flx.Label(wrap=1, text='Segmented Masks Folder Name')
                    self.TrackerMasks = flx.LineEdit(placeholder_text='Folder Name',
                                                     text='Example')

                    flx.Label(wrap=1, text='Mask Image Type')
                    self.TrackerMaskType = flx.LineEdit(placeholder_text='File Type',
                                                        text='png')

                    flx.Label(wrap=1, text='Tracked Image Save Type')
                    self.TrackedSaveType = flx.LineEdit(placeholder_text='File Type',
                                                        text='png')

                    flx.Label(wrap=1, text='CSV File Name')
                    self.TrackedCSVSave = flx.LineEdit(placeholder_text='File Name',
                                                       text='Data')

                    flx.Label(wrap=1, text='Minimum Pixel Sharing To Be Considered Same Object')
                    self.TrackedMinPixelShare = flx.LineEdit(placeholder_text='Percent', text='0.5')

                    flx.Label(wrap=1, text='Frame Range To Correct Flickering')
                    self.TrackedFrameRange = flx.LineEdit(placeholder_text='Frame Range', text='2')

                    flx.Label(wrap=1, text='Minimum Pixel Sharing To Be Considered Merge')
                    self.TrackedMerge = flx.LineEdit(placeholder_text='Frame Range', text='0.5')

                    flx.Label(wrap=1, text='Minimum Pixel Sharing To Be Considered Split')
                    self.TrackedSplit = flx.LineEdit(placeholder_text='Frame Range', text='0.5')

                    flx.Label(wrap=1, text='Minimum Lifespan')
                    self.TrackedLife = flx.LineEdit(placeholder_text='Frame Range', text='1')

                    flx.Label(wrap=1, text='Minimum size')
                    self.TrackedSize = flx.LineEdit(placeholder_text='Frame Range', text='4')

                ###############################################################################
                # Analyzing
                ###############################################################################
                with flx.VBox(style='border:1px solid #777;'):
                    flx.Label(text='Analysis')

            self.but_start = flx.Button(text='Start')

    @flx.reaction('but_train_settings.pointer_click')
    def TrainAdvSettings(self, *events):
        if self.showingCheck==False:
            self.temp.apply_style("visibility: visible;")
            self.showingCheck = True
        else:
            self.temp.apply_style("visibility: hidden;")
            self.showingCheck = False

    @flx.reaction('but_start.pointer_click')
    def Start(self, *events):
        #Gather all the needed variables
        #Check to see if we need to train
        if self.TrainBox.checked:
            #If Train is checked, lets process the associated variables
            TrainName = self.TrainName.text
            TrainPath = self.TrainPath.text
            NumChannles = self.NumChannles.text
            ImgType = self.ImgType.text
            MaskType = self.MaskType.text
            ImageWidthHeight = self.ImageWidthHeight.text
            MinPercentWhite = self.MinPercentWhite.text
            ValidationSplit = self.ValidationSplit.text
            SmallKernal = self.SmallKernal.text
            SmallBatch = self.SmallBatch.text
            SmallEpochs = self.SmallEpochs.text
            SmallThresh = self.SmallThresh.text
            BoundaryKernal = self.BoundaryKernal.text
            BoundaryW0 = self.BoundaryW0.text
            BoundarySigma = self.BoundarySigma.text
            BoundaryBatch = self.BoundaryBatch.text
            BoundaryEpochs = self.BoundaryEpochs.text
            BoundaryThresh = self.BoundaryThresh.text
            MixerKernal = self.MixerKernal.text
            MixerBatch = self.MixerBatch.text
            MixerEpochs = self.MixerEpochs.text

        #Check to see if we need to segment
        if self.SegBox.checked:
            #If Segment is checked, lets process the associated variables
            SegmentName=self.SegmentName.text
            SegmentImages=self.SegmentImages.text
            SegmentImgType=self.SegmentImgType.text
            SegmentMaskSave=self.SegmentMaskSave.text
            SegmentMaskType=self.SegmentMaskType.text
            SegmentThresh=self.SegmentThresh.text

        #Check to see if we need to track
        if self.TrackBox.checked:
            # If Segment is checked, lets process the associated variables
            TrackerImages=self.TrackerImages.text
            TrackerImageType=self.TrackerImageType.text
            TrackerMasks=self.TrackerMasks.text
            TrackerMaskType=self.TrackerMaskType.text
            TrackedSaveType=self.TrackedSaveType.text
            TrackedCSVSave=self.TrackedCSVSave.text
            TrackedMinPixelShare=self.TrackedMinPixelShare.text
            TrackedFrameRange=self.TrackedFrameRange.text
            TrackedMerge=self.TrackedMerge.text
            TrackedSplit=self.TrackedSplit.text
            TrackedLife=self.TrackedLife.text
            TrackedSize=self.TrackedSize.text

        # Check to see if we need to analyze
        if self.AnalyzeBox.checked:
            # If Segment is checked, lets process the associated variables
            print('HEY')


        #Run the Requested Code
        if self.TrainBox.checked:
            train(TrainName, TrainPath, NumChannles, ImgType, MaskType, int(ImageWidthHeight),
                  float(MinPercentWhite), float(ValidationSplit), int(SmallKernal),int(SmallBatch),
                  int(SmallEpochs), float(SmallThresh), int(BoundaryKernal), int(BoundaryW0),
                  int(BoundarySigma), int(BoundaryBatch), int(BoundaryEpochs), float(BoundaryThresh),
                  int(MixerKernal), int(MixerBatch), int(MixerEpochs))

        if self.SegBox.checked:
            segment(SegmentImgType, SegmentMaskType, SegmentImages, SegmentMaskSave,
                    SegmentName, float(SegmentThresh))

        if self.TrackBox.checked:

            track(TrackerImageType, TrackedSaveType, TrackerMaskType, TrackedCSVSave, TrackerMasks,
                         TrackerImages, float(TrackedMinPixelShare), int(TrackedFrameRange),
                         float(TrackedSplit), float(TrackedMerge), int(TrackedLife), float(TrackedSize))

        if self.AnalyzeBox.checked:
            print('HEY')

app = flx.App(TuningFork)
# app.export('Tuning Fork.html', link=0)  # Export to single file
app.launch('app')
flx.run()  # enter the mainloop