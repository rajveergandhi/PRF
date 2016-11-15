import math
import numpy as np
from scipy import ndimage as ndi, stats


class TextureAnalysis():

    def __init__(self):
        self.name = "Texture Analysis Function"
        self.description = "Applied to quantify texture within a raster."

    def getParameterInfo(self):
        return [
            {
                'name': 'raster',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Input Raster",
                'description': "Input raster for texture analysis",
            },
            {
                'name': 'mode',
                'dataType': 'string',
                'value': 'Variance',
                'required': True,
                'domain': ('Variance', 'Skewness', 'Kurtosis'),
                'displayName': "Texture Analysis Operator",
                'description': "Mathematical formula for texture analysis."
            },
            {
                'name': 'win',
                'dataType': 'numeric',
                'value': 3,
                'required': True,
                'displayName': "Moving Window Size",
                'description': "Enter size of square data window for computation of local mean and deviation."
                               "Must be an odd number. If you enter an even number, function automatically adds one (+1) to make the number odd."
            },
        ]

    def getConfiguration(self, **scalars):
        return {
          'inheritProperties': 4 | 8,           # inherit everything but the pixel type (1) and NoData (2)
          'invalidateProperties': 2 | 4 | 8,    # invalidate these aspects because we are modifying pixel values and updating key properties.
          'padding': 0,                         # no padding to the input pixel block.
          'inputMask': False                    # input mask in .updatePixels() not required.
         }

    def updateRasterInfo(self, **kwargs):
        
        windowSizeTemp = kwargs['win'] 
        self.windowSize = windowSizeTemp if (windowSizeTemp % 2 != 0) else windowSizeTemp + 1

        mode = kwargs['mode'].lower()
        if mode == "variance": self.op = ndi.variance
        elif mode == "skewness": self.op = stats.skew
        else: self.op = stats.kurtosis

        self.bandCount = kwargs['raster_info']['bandCount']
                
        # output raster information
        kwargs['output_info']['bandCount'] = kwargs['raster_info']['bandCount']
        kwargs['output_info']['pixelType'] = 'u2'
        kwargs['output_info']['statistics'] = ()
        kwargs['output_info']['histogram'] = ()
        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):

        r = np.array(pixelBlocks['raster_pixels'], dtype='f4', copy=False)
        outBlock = np.zeros_like(r)

        for band in xrange(self.bandCount):
            outBlock[band] = ndi.generic_filter(r[band], self.op, (self.windowSize,self.windowSize))
        pixelBlocks['output_pixels'] = outBlock.astype(props['pixelType'], copy=False)

        return pixelBlocks

    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:                             # dataset-level properties
            keyMetadata['datatype'] = 'Processed'       # outgoing dataset is now 'Processed'
        return keyMetadata

# ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ##
