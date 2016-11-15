import numpy as np
from skimage import restoration
import math

class nlMeans():

    def __init__(self):
        self.name = "Non Local Means Filter"
        self.description = "A filter which is used for denoising images with specific textures."

    def getParameterInfo(self):
        return [
            {
                'name': 'raster',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Input Raster",
                'description': "A single-band input raster for applying the Gamma Filter."
            },
            {
                'name': 'patch_size',
                'dataType': 'numeric',
                'value': 7,
                'required': True,
                'displayName': "Patch Size",
                'description': "Size of patches used for denoising."
            },
            {
                'name': 'patch_distance',
                'dataType': 'numeric',
                'value': 5,
                'required': True,
                'displayName': "Patch Distance",
                'description': "Maximal distance in pixels where to search patches used for denoising."
            },
            {
                'name': 'cutOffDistance',
                'dataType': 'numeric',
                'value': 0.1,
                'required': True,
                'displayName': "Cut-off distance",
                'description': "The higher h, the more permissive one is in accepting patches. A higher h results in a smoother image, at the expense of blurring features. For a Gaussian noise of standard deviation sigma, a rule of thumb is to choose the value of h to be sigma of slightly less."
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
        self.patch_size = int(kwargs.get('patch_size'))
        self.patch_distance = int(kwargs.get('patch_distance'))
        self.cutOff_distance = float(kwargs.get('cutOffDistance',0.1))
        kwargs['output_info']['pixelType'] = kwargs['raster_info']['pixelType']
        kwargs['output_info']['statistics'] = ()
        kwargs['output_info']['histogram'] = ()
        kwargs['output_info']['colormap'] = ()

        return kwargs


    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        r = np.array(pixelBlocks['raster_pixels'], dtype='f4', copy=False)
        outBlock = pixelBlocks['raster_pixels'].astype(props['pixelType'],copy=False)

        for band in xrange(self.bandCount):
            rBand = r[band]
            row = rBand.shape[0]
            col = rBand.shape[1]
            outBlock[band] = restoration.denoise_nl_means(rBand, self.patch_size, self.patch_distance, self.cutOff_distance)

        pixelBlocks['output_pixels'] = outBlock.astype(props['pixelType'], copy=False)
        return pixelBlocks


    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:                             # dataset-level properties
            keyMetadata['datatype'] = 'Processed'       # outgoing dataset is now 'Processed'
        elif bandIndex == 0:                            # properties for the first band
            keyMetadata['wavelengthmin'] = None         # reset inapplicable band-specific key metadata
            keyMetadata['wavelengthmax'] = None
            keyMetadata['bandname'] = 'nlMeans'
        return keyMetadata

# ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ##

"""
references:

    .. [1] Buades, A., Coll, B., & Morel, J. M. (2005, June). A non-local
           algorithm for image denoising. In CVPR 2005, Vol. 2, pp. 60-65, IEEE.

    .. [2] J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen, Fast
           nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
           International Symposium on Biomedical Imaging: From Nano to Macro,
           2008, pp. 1331-1334.

    .. [3] Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
           Denoising. Image Processing On Line, 2014, vol. 4, p. 300-326.

"""