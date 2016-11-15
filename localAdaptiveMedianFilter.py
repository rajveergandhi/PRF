import numpy as np
from scipy import signal
import math

class localAdaptiveMedianFilter():

    def __init__(self):
        self.name = "Sobel Filter"
        self.description = "Reduces degradation and noise in an image using Wiener Filter."

    def getParameterInfo(self):
        return [
            {
                'name': 'raster',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Input Raster",
                'description': "A single-band input raster for applying the Gabor Filter."
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
          'inheritProperties':  4 | 8,              # inherit everything but the pixel type (1) and NoData (2)
          'invalidateProperties': 2 | 4 | 8,        # invalidate these aspects because we are modifying pixel values and updating key properties.
          'padding': 0,                             # no padding of the input pixel block
          'inputMask': False                        # we don't need the input mask in .updatePixels()
        }

    def updateRasterInfo(self, **kwargs):

        kwargs['output_info']['bandCount'] = kwargs['raster_info']['bandCount']
        self.bandCount = kwargs['raster_info']['bandCount']
        windowSize = int(kwargs.get('win', 3))
        self.windowSize = windowSize if (windowSize % 2 != 0) else windowSize + 1  # moving window size
        kwargs['output_info']['pixelType'] = kwargs['raster_info']['pixelType']
        kwargs['output_info']['statistics'] = ()
        kwargs['output_info']['histogram'] = ()
        kwargs['output_info']['colormap'] = ()

        return kwargs


    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        r = np.array(pixelBlocks['raster_pixels'], dtype ='f4', copy=False)
        outBlock = pixelBlocks['raster_pixels'].astype(props['pixelType'],copy=False)
        row = r.shape[1]
        col = r.shape[2]
        win_offset = int(self.windowSize/2)

        for band in xrange(self.bandCount):
            for i in xrange(row):
                xleft = i - win_offset
                xright = i + win_offset
                if xleft < 0:
                    xleft = 0
                if xright >= row:
                    xright = row
                for j in xrange(col):
                    yup = j - win_offset
                    ydown = j + win_offset
                    if yup < 0:
                        yup = 0
                    if ydown >= col:
                        ydown = col

                    pix_value = r[band][i, j]
                    window = r[band][xleft:xright, yup:ydown]
                    new_pix_value = np.median(window)
                    outBlock[band][i, j] = np.round(new_pix_value)


        pixelBlocks['output_pixels'] = outBlock.astype(props['pixelType'], copy=False)
        return pixelBlocks


    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:                             # dataset-level properties
            keyMetadata['datatype'] = 'Processed'       # outgoing dataset is now 'Processed'
        elif bandIndex == 0:                            # properties for the first band
            keyMetadata['wavelengthmin'] = None         # reset inapplicable band-specific key metadata
            keyMetadata['wavelengthmax'] = None
            keyMetadata['bandname'] = 'WienerFilter'
        return keyMetadata

# ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ##

"""
References:
    [1]. Wikipedia: Wiener Deconvolution .
    http://en.wikipedia.org/wiki/Wiener_deconvolution
"""