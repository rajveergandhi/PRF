import numpy as np
from scipy import ndimage as ndi
from scipy import signal
import math

class GaborFilter2():

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
                'value': 1.0,
                'required': True,
                'displayName': "Psi Value",
                'description': "Phase offset."
            },
            {
                'name': 'lambda',
                'dataType': 'numeric',
                'value': 4.0,
                'required': True,
                'displayName': "Lambda Value",
                'description': "The wavelength of the sinusoidal factor."
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
                'value': 0.3,
                'required': True,
                'displayName': "Gamma Value",
                'description': "Spatial aspect ratio. Specifies the ellipticity of the support of the Gabor function."
            },
        ]

    def getConfiguration(self, **scalars):
        return {
          'inheritProperties':  1 | 2 | 4 | 8,  # inherit all properties
          'invalidateProperties': 2 | 4 | 8,    # invalidate these aspects because we are modifying pixel values and updating key properties.
          'padding': 0,                         # no padding of the input pixel block
          'inputMask': False                    # we don't need the input mask in .updatePixels()
        }

    def updateRasterInfo(self, **kwargs):
        self.theta = kwargs.get('theta', 0.0)
        self.lamba = kwargs.get('lambda', 4.0)
        self.psi   = kwargs.get('psi', 1.0)
        self.sigma = kwargs.get('sigma', 2.0)
        self.gamma = kwargs.get('gamma', 0.3)
        self.nstds = 3       ## standard bounding box

        self.bandCount = kwargs['raster_info']['bandCount']
        if self.bandCount > 1:
            raise Exception('Image must be 1 band.')

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
        y, x = np.mgrid[0:row, 0:col]
        xTemp = x * np.cos(self.theta) + y * np.sin(self.theta)
        yTemp = -x * np.sin(self.theta) + y * np.cos(self.theta)
        weight = np.exp(-1 * ((xTemp ** 2 + self.gamma ** 2 * yTemp ** 2)) / (2 * self.sigma ** 2)) * np.cos(2 * np.pi  * xTemp / self.lamba + self.psi)

        weight = weight.reshape((self.bandCount,row,col))

        g = np.zeros(raster.shape, dtype=np.double)
        g[:] = weight
        g[:] = g / np.sum(g)

        if self.bandCount == 1:
            outBlock = ndi.convolve(raster, g)
            pixelBlocks['output_pixels'] = raster.astype(props['pixelType'], copy=False)

        else:
            outBlockMultiband = pixelBlocks['raster_pixels'].astype(props['pixelType'],copy=False)
            for band in xrange(self.bandCount):
                outBlockMultiband[band] = ndi.convolve(raster[band], g)
            pixelBlocks['output_pixels'] = raster.astype(props['pixelType'], copy=False)

        return pixelBlocks


    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:                             # dataset-level properties
            keyMetadata['datatype'] = 'Processed'       # outgoing dataset is now 'Processed'
        elif bandIndex == 0:                            # properties for the first band
            keyMetadata['wavelengthmin'] = None         # reset inapplicable band-specific key metadata
            keyMetadata['wavelengthmax'] = None
            keyMetadata['bandname'] = 'GaborFilter'
        return keyMetadata
