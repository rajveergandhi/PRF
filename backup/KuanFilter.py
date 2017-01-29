import math
import numpy as np
from scipy import ndimage as ndi, stats
from scipy.signal import medfilt
from scipy.stats import variation
from numpy import ndarray

class KuanFilter():

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

        self.bandCount = kwargs['raster_info']['bandCount']
                
        # output raster information
        kwargs['output_info']['bandCount'] = kwargs['raster_info']['bandCount']
        kwargs['output_info']['pixelType'] = 'u2'
        kwargs['output_info']['statistics'] = ()
        kwargs['output_info']['histogram'] = ()
        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):

        r = np.array(pixelBlocks['raster_pixels'], dtype='f4', copy=False)
        cu = np.float(0.25)
        row = r.shape[1]
        col = r.shape[2]
        win_offset = int(self.windowSize/2)
        outBlockMultiband = np.zeros_like(r)

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

                    two_cu = cu * cu
                    ci = np.std(window)/np.mean(window)
                    two_ci = ci * ci
                    if not two_ci:
                        two_ci = 0.1
                    divisor = 1.0 + two_cu
                    if cu > ci:
                        w_t = 0.0
                    else:
                        w_t = (1.0 - (two_cu / two_ci)) / divisor

                    a = str(w_t)
                    window_mean = window.mean()
                    b = str(window_mean)
                    new_pix_value = (pix_value * w_t) + (window_mean * (1.0 - w_t))
                    c = str(new_pix_value)

                    outBlockMultiband[band][i, j] = np.round(new_pix_value)

            pixelBlocks['output_pixels'] = outBlockMultiband.astype(props['pixelType'], copy=False)

        return pixelBlocks

    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:                             # dataset-level properties
            keyMetadata['datatype'] = 'Processed'       # outgoing dataset is now 'Processed'
        elif bandIndex == 0:                            # properties for the first band
            keyMetadata['wavelengthmin'] = None         # reset inapplicable band-specific key metadata
            keyMetadata['wavelengthmax'] = None
            keyMetadata['bandname'] = 'KuanFilter'
        return keyMetadata

# ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ##
"""
References:
    Speckle Contrast Image Analysis
    http://www.utdallas.edu/~ffqiu/published/2004Qiu-GISRS.pdf
"""