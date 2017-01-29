import numpy as np
import math
from scipy import stats

class ChangeDetection():

    def __init__(self):
        self.name = "Change Detection Function"
        self.description = "Apply Change Detection Algorithms to analyze temporal effects of multi-temporal datasets."

    def getParameterInfo(self):
        return [
            {
                'name': 'raster',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Raster A",
                'description': "Temporal raster dataset for Change Detection."
            },
            { 
                'name': 'sb1', 
                'dataType': 'numeric', 
                'value': 1, 
                'required': True, 
                'displayName': "Source Band 1", 
                'description': "The index of the first source band. The first band has index 1." 
            }, 
            { 
                'name': 'sb2', 
                'dataType': 'numeric', 
                'value': 2, 
                'required': True, 
                'displayName': "Source Band 2", 
                'description': "The index of the second source band. The second band has index 2." 
            }, 
            {
                'name': 'mode',
                'dataType': 'string',
                'value': 'Differencing',
                'required': True,
                'domain': ('Differencing', 'Ratioing', 'Logarithmic'),
                'displayName': "Change Detection Technique",
                'description': "Select technique to apply Change Detection Algorithm."
            },
        ]

    def getConfiguration(self, **scalars):
        self.sb1 = int(scalars.get('sb1', 1))
        self.sb2 = int(scalars.get('sb2', 2))

        return {
          'inheritProperties': 4 | 8,         # inherit everything but the pixel type (1) and NoData (2)
          'invalidateProperties': 2 | 4 | 8,  # invalidate these aspects because we are modifying pixel values and updating key properties.
          'padding': 0,                       # no padding on each of the input pixel block
          'inputMask': False                  # we don't need the input mask in .updatePixels()
        }

    def updateRasterInfo(self, **kwargs):
        self.mode = kwargs['mode']
        self.bandCount = kwargs['raster_info']['bandCount']
        if self.bandCount == 1:
            raise Exception('Image must be more than 1 band.')

        # output raster information
        kwargs['output_info']['bandCount'] = 1
        kwargs['output_info']['statistics'] = ()
        kwargs['output_info']['histogram'] = ()
        kwargs['output_info']['pixelType'] = 'u2'
        return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        r = np.array(pixelBlocks['raster_pixels'], dtype='f4', copy=False)
        sb1 = np.array(r[self.sb1-1])
        sb2 = np.array(r[self.sb2-1])

        # apply change detection algorithm
        if self.mode == "Differencing": t = sb1 - sb2
        elif self.mode == "Ratioing":   t = sb1 / sb2
        elif self.mode == "Logarithmic": t = np.log(np.maximum((sb1/sb2), 1.e-15))

        # output pixel block with change detection applied
        pixelBlocks['output_pixels'] = t.astype(props['pixelType'], copy=False)
        return pixelBlocks

    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:                             # dataset-level properties
            keyMetadata['datatype'] = 'Scientific'      # outgoing dataset is now 'Scientific'
        elif bandIndex == 0:                            # properties for the first band
            keyMetadata['wavelengthmin'] = None         # reset inapplicable band-specific key metadata
            keyMetadata['wavelengthmax'] = None
            keyMetadata['bandname'] = 'ChangeDetection'
        return keyMetadata

# ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ##

