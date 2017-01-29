import numpy as np
from scipy import ndimage as ndi
from scipy import signal
import math
from skimage.filters import gabor

class GaborFilter():

    def __init__(self):
        self.name = "Gabor Filter"
        self.description = "Extract texture features using the Gabor Filter."

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
                'name': 'theta',
                'dataType': 'numeric',
                'value': 0.0,
                'required': True,
                'displayName': "Theta Value",
                'description': "Orientation of the normal to the parallel stripes of a Gabor function."
            },
            {
                'name': 'psi',
                'dataType': 'numeric',
                'value': 0.0,
                'required': True,
                'displayName': "Psi Value",
                'description': "Phase offset."
            },
            {
                'name': 'frequency',
                'dataType': 'numeric',
                'value': 0.1,
                'required': True,
                'displayName': "Frequency",
                'description': "Spatial frequency of the harmonic function. Specified in pixels."
            },
            {
                'name': 'sigma',
                'dataType': 'numeric',
                'value': 2.0,
                'required': True,
                'displayName': "Sigma Value",
                'description': "The sigma/standard deviation of the Gaussian envelope."
            },
            {
                'name': 'gamma',
                'dataType': 'numeric',
                'value': 0.5,
                'required': True,
                'displayName': "Gamma Value",
                'description': "Spatial aspect ratio. Specifies the ellipticity of the support of the Gabor function."
            },
            {
                'name': 'n_stds',
                'dataType': 'numeric',
                'value': 3,
                'required': True,
                'displayName': "Number of standard Deviations",
                'description': "The linear size of the kernel."
            },
        ]

    def getConfiguration(self, **scalars):
        return {
          'inheritProperties': 4 | 8,           # inherit all properties
          'invalidateProperties': 2 | 4 | 8,    # invalidate these aspects because we are modifying pixel values and updating key properties.
          'padding': 0,                         # no padding of the input pixel block
          'inputMask': False                    # we don't need the input mask in .updatePixels()
        }

    def updateRasterInfo(self, **kwargs):
        self.theta = kwargs.get('theta', 0.0)
        self.frequency = kwargs.get('frequency', 0.1)
        self.psi   = kwargs.get('psi', 0.0)
        self.sigma = kwargs.get('sigma', 2.0)
        self.gamma = kwargs.get('gamma', 0.5)
        self.n_stds = kwargs.get('n_stds', 3)

        self.bandCount = kwargs['raster_info']['bandCount']

        kwargs['output_info']['bandCount'] = kwargs['raster_info']['bandCount']
        kwargs['output_info']['pixelType'] = 'u2'
        kwargs['output_info']['statistics'] = ()
        kwargs['output_info']['histogram'] = ()
        kwargs['output_info']['colormap'] = ()

        return kwargs


    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        raster = np.array(pixelBlocks['raster_pixels'], dtype ='f4', copy=False)
        row = raster.shape[1]
        col = raster.shape[2]

        #CREATE GABOR KERNEL
        sigma_x = self.sigma
        sigma_y = self.sigma / self.gamma

        outBlockMultiband = pixelBlocks['raster_pixels'].astype(props['pixelType'],copy=False)
        for band in xrange(self.bandCount):
            gReal, gImag = gabor(raster[band], self.frequency, self.theta, 1, sigma_x, sigma_y,self.n_stds, self.psi, mode='reflect', cval=0)
            outBlockMultiband[band] = gReal
        pixelBlocks['output_pixels'] = outBlockMultiband.astype(props['pixelType'], copy=False)

        return pixelBlocks


    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:                             # dataset-level properties
            keyMetadata['datatype'] = 'Processed'       # outgoing dataset is now 'Processed'
        elif bandIndex == 0:                            # properties for the first band
            keyMetadata['wavelengthmin'] = None         # reset inapplicable band-specific key metadata
            keyMetadata['wavelengthmax'] = None
            keyMetadata['bandname'] = 'GaborFilter'
        return keyMetadata

# ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ##

"""
References:
    [1]. Wikipedia: Gabor filter .
    http://en.wikipedia.org/wiki/Gabor_filter
"""