import math
import numpy as np
from scipy import ndimage as ndi

class Wallis():

    def __init__(self):
        self.name = "Wallis Normalization Function"
        self.description = "Normalize pixel value for given image."

    def getParameterInfo(self):
        return [
            {
                'name': 'raster',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Input Raster",
                'description': "grayscale raster for texture analysis.",
            },
            {
                'name': 'dsmean',
                'dataType': 'numeric',
                'value': 128.0,
                'required': True,
                'displayName': "Desired Mean",
                'description': "Enter desired mean for the output image. Min value: 0 Max value: 255"
            },
            {
                'name': 'dsstd',
                'dataType': 'numeric',
                'value': 76.8,
                'required': True,
                'displayName': "Desired Standard Deviation",
                'description': "Enter desired standard deviation for the output image. Min value: 1 Max value:255"
            },
            {
                'name': 'win',
                'dataType': 'numeric',
                'value': 21,
                'required': True,
                'displayName': "Moving Window Size",
                'description': "Enter size of square data window for computation of local mean and deviation."
                               "Must be an odd number. If you enter an even number, function automatically adds one (+1) to make the number odd."
            },
            {
                'name': 'gain',
                'dataType': 'numeric',
                'value': 6.0,
                'required': True,
                'displayName': "Contrast Expansion (Gain)",
                'description': "Enter maximum contrast expansion (Gain). Min value: 0 Max value: 255"
            },
            {
                'name': 'alpha',
                'dataType': 'numeric',
                'value': 0.8,
                'required': True,
                'displayName': "Brightness Forcing Constant",
                'description': "Enter factor to govern mean value shifting. Min value: 0"
            },
        ]

    def getConfiguration(self, **scalars):
        return {
          'inheritProperties': 2 | 4 | 8,       # inherit everything but the pixel type (1)
          'invalidateProperties': 2 | 4 | 8,    # invalidate these aspects because we are modifying pixel values and updating key properties.
          'padding': 0,                         # no padding of the input pixel block
          'inputMask': False                    # we don't need the input mask in .updatePixels()
        }

    def updateRasterInfo(self, **kwargs):
        # output raster information
        kwargs['output_info']['bandCount'] = kwargs['raster_info']['bandCount']
        kwargs['output_info']['pixelType'] = 'u2'

        # input parameters for function
        self.bandCount = kwargs['raster_info']['bandCount']
        self.dsMean = kwargs.get('dsmean', 128.0)  # desired mean
        self.dsStd = kwargs.get('dsstd', 76.8)  # desired deviation
        windowSize = int(kwargs.get('win', 3))
        self.windowSize = windowSize if (windowSize % 2 != 0) else windowSize + 1  # moving window size
        self.maxGain = kwargs.get('gain', 6.0)  # maximum gain
        self.alpha = kwargs.get('alpha', 0.8)  # alpha

        # check range of parameters
        if (self.dsMean > 255.0 or self.dsMean < 0.0) or (self.dsStd > 255.0 or self.dsStd < 1.0) or \
           (self.maxGain > 255.0 or self.maxGain < 0.0) or (self.alpha < 0.0):
            raise Exception("Input parameter out of range.")

        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        # get input raster for applying wallis filter
        r = np.array(pixelBlocks['raster_pixels'], dtype='f4', copy=False)
        outBlock = pixelBlocks['raster_pixels'].astype(props['pixelType'],copy=False)

        for band in xrange(self.bandCount):
            rBand = r[band]
            row = rBand.shape[0]
            col = rBand.shape[1]

            rBand = rBand.reshape((1,row,col))

            # calculate local statistics
            localmean = ndi.uniform_filter(rBand, self.windowSize)
            localstddev = ndi.generic_filter(rBand, np.std, self.windowSize)

            # apply wallis normalization filter, output pixel array with wallis filter applied
            outBlockBand = np.sqrt(self.dsStd/((1 / self.maxGain) + localstddev)) * (rBand - localmean) + (self.alpha * self.dsMean) + ((1 - self.alpha) * localmean) 
            
            outBlock[band] = outBlockBand.reshape((row,col))


        pixelBlocks['output_pixels'] = outBlock.astype(props['pixelType'], copy=False)
        return pixelBlocks

    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:                             # dataset-level properties
            keyMetadata['datatype'] = 'Processed'       # outgoing dataset is now 'Processed'
        return keyMetadata

# ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ##
"""
References:
    Wallis Normalization Image Enhancement Tactical Support System (TESS (3)) Documentation
    http://www.dtic.mil/dtic/tr/fulltext/u2/a248301.pdf
"""