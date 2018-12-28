from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog



class deepLabV3_GUI:
    TestImage=[]
    SegmentedImage=[]
    def __init__(self,master):
        master.title('Semantic Segmentation : Deep Lab V3')
        master.minsize(200, 200)

        #Setting Up Test Image Panel

        self.testImgFrame = Frame(master,highlightbackground="green",
                             highlightcolor="green",
                             highlightthickness=2).\
                                 grid(row=0,column=0)

        self.testImgLabel = Label(self.testImgFrame,
                                  image = self.TestImage,
                                  text = 'Test Image Will Be Loaded Here',
                                  relief=SUNKEN)
        self.testImgLabel.grid(row=0,column=0)

        self.loadtestImg = Button(self.testImgFrame,
                                  text='Load Test Image', 
                                  command=self.loadTestImage).\
                                       grid(row=1, column=0,sticky=W)


        #Setting Up Segmented Image Panel

        self.userCommandsFrame = Frame(master,highlightbackground="green",
                                  highlightcolor="green",
                                  highlightthickness=2).\
                                      grid(row=1,column=2)

        self.segmentedImgLabel = Label(self.userCommandsFrame,
                                       image = self.SegmentedImage,
                                       text = 'Segmented Image Will Be Displayed Here',
                                       relief=SUNKEN)
        self.segmentedImgLabel.grid(row=0,column=2)

        self.restoreWeightsButton = Button(self.userCommandsFrame,
                                           text='Restore Model Weights',
                                           command=self.restoreModelWeights).\
                                               grid(row=1, column=2,sticky=W)

        self.segmentTestImage = Button(self.userCommandsFrame,
                                           text='Segment Test Image',
                                           command=self.segmentTestImage).\
                                               grid(row=2, column=2,sticky=W)


        #Quit Button
        self.quitButton = Button(self.userCommandsFrame,
                                 text='Quit',
                                 command=master.destroy).\
                                     grid(row=3, column=2, sticky=W)


    def restoreModelWeights(self):
        print('so it works')
    
    def loadTestImage(self):
        path=filedialog.askopenfilename(filetypes=[("Image Format",'.jpg'),("Image Format",'.png')])
        self.TestImage = Image.open(path)
        tkimage = ImageTk.PhotoImage(self.TestImage)
        self.testImgLabel.configure(image=tkimage)
        self.testImgLabel.image=tkimage
        self.testImgLabel.grid(row=0,column=0)
        #TestImage.show()

    def segmentTestImage(self):
        # self.SegmentedImage = ImplementThisFunctionForSegmentation(self.TestImage)
        tkimage = ImageTk.PhotoImage(self.TestImage)
        self.segmentedImgLabel.configure(image=tkimage)
        self.segmentedImgLabel.image=tkimage
        self.segmentedImgLabel.grid(row=0,column=2)

mainWindow = Tk()
deepLabV3 = deepLabV3_GUI(mainWindow)
mainWindow.mainloop()