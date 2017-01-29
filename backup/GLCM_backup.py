from scipy import ndimage
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import math
import utils

class GLCM2():

    def __init__(self):
        self.name = "GLCM Function"
        self.description = ""
        self.trace = utils.Trace()

    def getParameterInfo(self):
        return [
            {
                'name': 'raster',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Input Raster",
                'description': "",
            },
            {
                'name': 'output',
                'dataType': 'string',
                'value': "Contrast",
                'required': True,
                'domain': ('Contrast', 'Dissimilarity', 'Homogeneity','Angular Second Moment', 'Energy', \
                           'GLCM Mean', 'GLCM Variance', 'GLCM Correlation'),
                'displayName': "Output Statistic",
                'description': "Output Feature of the Raster",
            },
            #{
            #    'name': 'win',
            #    'dataType': 'string',
            #    'value': "7x7",
            #    'required': True,
            #    'domain': ('5x5', '7x7', '9x9','11x11','3x3'),
            #    'displayName': "Window Size",
            #    'description': "",
            #},
            {
                'name': 'angle',
                'dataType': 'string',
                'value': "ALL",
                'required': True,
                'domain': ('0', '45', '90','135', 'ALL'),
                'displayName': "Angle",
                'description': "",
            },
            {
                'name': 'quantizer',
                'dataType': 'string',
                'value': "Probabilistic Quantizer",
                'required': True,
                'domain': ("Probabilistic Quantizer","Equal Distance Quantizer"),
                'displayName': "Quantizer",
                'description': "",
            },
            {
                'name': 'levels',
                'dataType': 'string',
                'value': "256",
                'required': True,
                'domain': ("2","4","8","16","32","64","128","256"),
                'displayName': "Quantization Levels",
                'description': "",
            },
            {
                'name': 'dist',
                'dataType': 'numeric',
                'value': 1,
                'required': True,
                'displayName': "Displacement",
                'description': "",
            },


        ]


    def getConfiguration(self, **scalars):
        return {
          'inheritProperties': 1 | 2 | 4 | 8,   # inherit everything but the pixel type (1) and NoData (2)
          'invalidateProperties': 2 | 4 | 8,    # invalidate these aspects because we are modifying pixel values and updating key properties.
          'padding': 0,                         # one extra on each each of the input pixel block
          'inputMask': False                    # we need the input mask in .updatePixels()
        }


    def updateRasterInfo(self, **kwargs):
        angle      = kwargs.get('angle','ALL')
        windowSize = kwargs.get('win','9x9')
        output     = kwargs.get('output','GLCM Mean')
        quantizer  = kwargs.get('quantizer',"Probabilistic Quantizer")
        levels     = kwargs.get('levels',"64")

        self.prepare(angle,windowSize,output,quantizer,levels)
        self.displacement = kwargs.get('dist',1)

        stats = kwargs['raster_info']['statistics']
        self.minimum = stats[0].get('minimum',0)      # Minimum pixel value of input raster
        self.maximum = stats[0].get('maximum',255)      # Maximum pixel value of input raster

        kwargs['output_info']['bandCount'] = 1
        kwargs['output_info']['pixelType'] = 'u2'
        kwargs['output_info']['statistics'] = ()
        kwargs['output_info']['histogram'] = ()
        kwargs['output_info']['colormap'] = ()

        return kwargs


    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        raster = np.array(pixelBlocks['raster_pixels'], dtype='u1', copy=False)
        row = raster.shape[1]
        col = raster.shape[2]
        
        ## apply linear stretch
        if self.maximum > 255 or self.minimum < 0:
            raster = (raster - self.minimum) * ((255)/(self.maximum - self.minimum)) + 0

        rasterBand = raster.reshape((row,col))

        # calculate grey level co-occurence matrix
        g = np.zeros((row,col),dtype=np.float)
        rasterBand = np.lib.pad(rasterBand,(3,3),'edge') 
        for i in xrange(row):
            for j in xrange(col):                
                #if self.windowSize == 7:
                #    if i <3 or j <3:
                #        continue
                #    if i > (rasterBand.shape[0] - 4) or j > (rasterBand.shape[0] - 4):
                #        continue                
                #    glcm_window = rasterBand[i-3: i+4, j-3 : j+4]
                #    glcm = greycomatrix(glcm_window,[self.displacement],[self.angle],int(self.levels),symmetric = True, normed=True)
                #    g[i][j] = greycoprops(glcm, self.output)
                #elif self.windowSize == 3:
                #    if i<1 or j<1:
                #        continue
                #    if i > (rasterBand.shape[0] - 2) or j > (rasterBand.shape[0] - 2):
                #        continue                
                #    glcm_window = rasterBand[i-1: i+2, j-1 : j+2]
                #    glcm = greycomatrix(glcm_window,[self.displacement],[self.angle],int(self.levels),symmetric = True, normed=True)
                #    g[i][j] = greycoprops(glcm, self.output)
                #elif self.windowSize == 5:
                #    if i<2 or j<2:
                #        continue
                #    if i > (rasterBand.shape[0] - 3) or j > (rasterBand.shape[0] - 3):
                #        continue                
                #    glcm_window = rasterBand[i-2: i+3, j-2 : j+3]
                #    glcm = greycomatrix(glcm_window,[self.displacement],[self.angle],int(self.levels),symmetric = True, normed=True)
                #    g[i][j] = greycoprops(glcm, self.output)
                glcm_window = rasterBand[i: i+2, j : j+2]
                glcm = greycomatrix(glcm_window,[self.displacement],[self.angle],int(self.levels),symmetric = True, normed=True)
                g[i][j] = greycoprops(glcm, self.output)

        xs = []
        xs.append(greycoprops(glcm,'energy')[0,0])
        xs.append(greycoprops(glcm,'correlation')[0,0])
        xs.append(greycoprops(glcm,'contrast')[0,0])
        xs.append(greycoprops(glcm,'dissimilarity')[0,0])
        self.trace.log("Trace|Cooccurrence.updatePixels.1|Energy : {0}|Correlation : {1}|Contrast : {2}|Dissimilarity : {3}\n".format(glcm,xs[1],xs[2],xs[3]))

        outBlock = g.reshape((1,row,col))
        pixelBlocks['output_pixels'] = outBlock.astype(props['pixelType'], copy=False)
        return pixelBlocks


    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:                             # dataset-level properties
            keyMetadata['datatype'] = 'Processed'       # outgoing dataset is now 'Processed'
        elif bandIndex == 0:                            # properties for the first band
            keyMetadata['wavelengthmin'] = None         # reset inapplicable band-specific key metadata
            keyMetadata['wavelengthmax'] = None
            keyMetadata['bandname'] = 'GLCM'
        return keyMetadata

    def prepare(self, angle, windowSize, output, quantizer, levels):
        ## assignment of windowSize
        if (windowSize == "5x5"):             self.windowSize = 5
        elif (windowSize == "9x9"):           self.windowSize = 9
        elif (windowSize == "7x7"):           self.windowSize = 7
        elif (windowSize == "3x3"):           self.windowSize = 3
        else:                                 self.windowSize = 11

        ## assignmnt of angle
        if (angle == "ALL"):                  self.angle = np.pi
        elif (angle == "0"):                  self.angle = 0
        elif (angle == "45"):                 self.angle = np.pi/8
        elif (angle == "90"):                 self.angle = np.pi/4
        else:                                 self.angle = np.pi/2

        ## assignment of output statistics
        if (output == "GLCM Mean"):                 self.output = 'mean'
        elif (output == "GLCM Variance"):           self.output = 'variance'
        elif (output == "GLCM Correlation"):        self.output = 'correlation'
        elif (output == "Contrast"):                self.output = 'contrast'
        elif (output == "Dissimilarity"):           self.output = 'dissimilarity'
        elif (output == "Homogeneity"):             self.output = 'homogeneity'
        elif (output == "Angular Second Moment"):   self.output = 'ASM'
        else:                                       self.output = 'energy'

        ## assignment of quantizer
        if (quantizer == "Probabilistic Quantizer"): self.quantizer = 0
        else:                                        self.quantizer = 1

        ## assignment of quantization levels
        if (levels == "2"):                 self.levels = 2
        elif (levels == "4"):               self.levels = 4
        elif (levels == "8"):               self.levels = 8
        elif (levels == "16"):               self.levels = 16
        elif (levels == "32"):               self.levels = 32
        elif (levels == "64"):               self.levels = 64
        elif (levels == "128"):               self.levels = 128
        else:                               self.levels = 256
