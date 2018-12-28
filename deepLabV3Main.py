from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog



class deepLabV3_GUI:
    TestImage=[]
    SegmentedImage=[]
    def __init__(self,master):
        master.title('Semantic Segmentation : Deep Lab V3')
        master.minsize(500, 200)

        self.testImgFrame = Frame(master,highlightbackground="green",
                             highlightcolor="green",
                             highlightthickness=2)
        self.testImgFrame.pack()
        self.testImgLabel = Label(self.testImgFrame,image = self.TestImage)
        self.loadtestImg = Button(self.testImgFrame,
                                  text='Load Test Image', 
                                  command=self.loadTestImage)
        self.loadtestImg.grid(row=0, column=0)


        self.userCommandsFrame = Frame(master,highlightbackground="green",
                                  highlightcolor="green",
                                  highlightthickness=2)
        self.userCommandsFrame.pack(side=BOTTOM)
        self.restoreWeightsButton = Button(self.userCommandsFrame,
                                           text='Restore Model Weights',
                                           command=self.restoreModelWeights)
        self.restoreWeightsButton.grid(row=0, column=0)

        self.quitButton = Button(self.userCommandsFrame,text='Quit',command=master.destroy)
        self.quitButton.grid(row=0, column=1)


    def restoreModelWeights(self):
        print('so it works')
    
    def loadTestImage(self):
        path=filedialog.askopenfilename(filetypes=[("Image Format",'.jpg'),("Image Format",'.png')])
        im = Image.open(path)
        tkimage = ImageTk.PhotoImage(im)
        self.testImgLabel.configure(image=tkimage)
        self.testImgLabel.image=tkimage
        self.testImgLabel.grid(row=1,column=0)
mainWindow = Tk()
deepLabV3 = deepLabV3_GUI(mainWindow)
mainWindow.mainloop()