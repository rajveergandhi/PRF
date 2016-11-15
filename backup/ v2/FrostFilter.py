import math
import numpy as np
from scipy import ndimage as ndi, stats
from scipy.signal import medfilt
from scipy.stats import variation
from numpy import ndarray

class FrostFilter():

    def __init__(self):
        self.name = "Frost Filter"
        self.description = "Use Frost filters to reduce speckle while preserving edges in radar images. The Frost filter is an exponentially damped circularly symmetric filter that uses local statistics. The pixel being filtered is replaced with a value calculated based on the distance from the filter center, the damping factor, and the local variance."

    def getParameterInfo(self):
        return [
            {
                'name': 'raster',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Input Raster",
                'description': "Input raster",
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
            {
                'name': 'damp',
                'dataType': 'numeric',
                'value': 2.0,
                'required': True,
                'displayName': "Damping Factor",
                'description': "The damping factor determines the amount of exponential damping and the default value of 1 is sufficient for most radar images. Larger damping values preserve edges better but smooth less, and smaller values smooth more. A damping value of 0 results in the same output as a low pass filter."
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

        self.dampFactor = kwargs['damp']

        self.bandCount = kwargs['raster_info']['bandCount']
                
        # output raster information
        kwargs['output_info']['bandCount'] = kwargs['raster_info']['bandCount']
        kwargs['output_info']['pixelType'] = 'u2'
        kwargs['output_info']['statistics'] = ()
        kwargs['output_info']['histogram'] = ()
        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):

        r = np.array(pixelBlocks['raster_pixels'], dtype='f4', copy=False)
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

                    window = r[band][xleft:xright, yup:ydown]
                    variation_coef = variation(window, None)
                    window_mean = window.mean()
                    sigma_zero = variation_coef / window_mean
                    factor_A = self.dampFactor * sigma_zero
                    window_flat = window.flatten()
                    weights_array = np.zeros(window.size)
                    N, M = window.shape
                    center_pixel = np.float64(window[N / 2, M / 2])
                    window_flat = np.float64(window_flat)
                    distances = np.abs(window_flat - center_pixel)

                    weights_array = np.exp(-factor_A * distances)
                    pixels_array = window.flatten()
                    weighted_values = weights_array * pixels_array

                    outBlockMultiband[band][i, j] = weighted_values.sum() / weights_array.sum()

            pixelBlocks['output_pixels'] = outBlockMultiband.astype(props['pixelType'], copy=False)

        return pixelBlocks

    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:                             # dataset-level properties
            keyMetadata['datatype'] = 'Processed'       # outgoing dataset is now 'Processed'
        elif bandIndex == 0:                            # properties for the first band
            keyMetadata['wavelengthmin'] = None         # reset inapplicable band-specific key metadata
            keyMetadata['wavelengthmax'] = None
            keyMetadata['bandname'] = 'FrostFilter'
        return keyMetadata

# ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ##