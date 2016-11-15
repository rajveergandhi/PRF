import math
import numpy as np
from sklearn.cluster import k_means,KMeans

class kMeans():

    def __init__(self):
        self.name = "K-Means Cluster Analysis Function"
        self.description = "Applies K-Means Clustering on input raster."

    def getParameterInfo(self):
        return [
            {
                'name': 'raster',
                'dataType': 'raster',
                'value': None,
                'required': True,
                'displayName': "Input Raster",
                'description': "The primary raster input."
            },
            {
                'name': 'cluster',
                'dataType': 'numeric',
                'value': 14,
                'required': True,
                'displayName': "Number of Clusters",
                'description': "The number of clusters to form as well as the number of centroids to generate."
            },
            {
                'name': 'iter',
                'dataType': 'numeric',
                'value': 30,
                'required': True,
                'displayName': "Number of Iterations (for single run)",
                'description': "Maximum number of iterations of the k-means algorithm for a single run."
            },
            {
                'name': 'n_init',
                'dataType': 'numeric',
                'value': 10,
                'required': True,
                'displayName': "Number of iterations (of k means algorithm)",
                'description': "Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia."
            },
        ]

    def getConfiguration(self, **scalars):
        return {
          'inheritProperties': 4 | 8,           # inherit everything but the pixel type (1) and NoData(2)
          'invalidateProperties': 2 | 4 | 8,    # invalidate these aspects because we are modifying pixel values(histogram, statistics) and updating key properties.
          'padding': 0,                         # no extra padding of the input pixel block
          'inputMask': False                    # we don't need the input mask in .updatePixels()
        }

    def updateRasterInfo(self, **kwargs):
        # assign input parameters
        self.cluster = kwargs['cluster']
        self.iter = kwargs['iter']
        self.n_init = int(kwargs['n_init'])

        # output raster information
        r = kwargs['raster_info']
        if r['bandCount'] == 4:
            kwargs['output_info']['bandCount'] = 4
            kwargs['output_info']['pixelType'] = 'u2'
            kwargs['output_info']['statistics'] = ({'minimum': 0, 'maximum': self.cluster - 1})
            return kwargs
        elif r['bandCount'] == 2 :
            kwargs['output_info']['bandCount'] = 2
            kwargs['output_info']['pixelType'] = 'u2'
            kwargs['output_info']['statistics'] = ({'minimum': 0, 'maximum': self.cluster - 1})
            return kwargs
        elif r['bandCount'] == 1 :
            kwargs['output_info']['bandCount'] = 1
            kwargs['output_info']['pixelType'] = 'u2'
            kwargs['output_info']['statistics'] = ({'minimum': 0, 'maximum': self.cluster - 1})
            return kwargs

    def updatePixels(self, tlc, shape, props, **pixelBlocks):
        # get the primary input raster    
        
        r = np.array(pixelBlocks['raster_pixels'], dtype='f4', copy=False)
        rStr = str(r)
        rStrShape = str(r.shape)

        if(np.shape(r)[0]==1):
            r3 = r.ravel()
            r4 = r3.reshape((r3.shape[0],1))

            km = KMeans(int(self.cluster),'k-means++',self.n_init,int(self.iter)).fit(r4)
            j = km.predict(r4)

            k = j.reshape((r.shape))
            kstr = str(k)

            pixelBlocks['output_pixels'] = k.astype(props['pixelType'], copy=False)
            return pixelBlocks

        elif(np.shape(r)[0]==2):
            first_band = np.array(r[0])
            second_band = np.array(r[1])
            fb_ravel = first_band.ravel()
            sb_ravel = second_band.ravel()
            fb = fb_ravel.reshape((fb_ravel.shape[0],1))
            sb = sb_ravel.reshape((sb_ravel.shape[0],1))
           
            km1 = KMeans(int(self.cluster),'k-means++',self.n_init,int(self.iter)).fit(fb)
            km2 = KMeans(int(self.cluster),'k-means++',self.n_init,int(self.iter)).fit(sb)
            j1 = km1.predict(fb)
            j2 = km2.predict(sb)

            k1 = j1.reshape((first_band.shape))
            k2 = j2.reshape((second_band.shape))

            pixelBlocks['output_pixels'] = pixelBlocks['raster_pixels'].astype(props['pixelType'],copy=False) 
            pixelBlocks['output_pixels'][0] = k1.astype(props['pixelType'], copy=False)
            pixelBlocks['output_pixels'][1] = k2.astype(props['pixelType'], copy=False)

            return pixelBlocks

        elif(np.shape(r)[0]==4):
            
            first_band = np.array(r[0])
            second_band = np.array(r[1])
            third_band = np.array(r[2])
            fourth_band = np.array(r[3])
            fb_ravel = first_band.ravel()
            sb_ravel = second_band.ravel()
            tb_ravel = third_band.ravel()
            fth_ravel = fourth_band.ravel()

            fb = fb_ravel.reshape((fb_ravel.shape[0],1))
            sb = sb_ravel.reshape((sb_ravel.shape[0],1))
            tb = tb_ravel.reshape((tb_ravel.shape[0],1))
            fthb = fth_ravel.reshape((fth_ravel.shape[0],1))

            km1 = KMeans(int(self.cluster),'k-means++',self.n_init,int(self.iter)).fit(fb)
            km2 = KMeans(int(self.cluster),'k-means++',self.n_init,int(self.iter)).fit(sb)
            km3 = KMeans(int(self.cluster),'k-means++',self.n_init,int(self.iter)).fit(tb)
            km4 = KMeans(int(self.cluster),'k-means++',self.n_init,int(self.iter)).fit(fthb)

            j1 = km1.predict(fb)
            j2 = km2.predict(sb)
            j3 = km3.predict(tb)
            j4 = km4.predict(fthb)

            k1 = j1.reshape((first_band.shape))
            k2 = j2.reshape((second_band.shape))
            k3 = j3.reshape((third_band.shape))
            k4 = j4.reshape((fourth_band.shape))

            pixelBlocks['output_pixels'] = pixelBlocks['raster_pixels'].astype(props['pixelType'],copy=False) 
            pixelBlocks['output_pixels'][0] = k1.astype(props['pixelType'], copy=False)
            pixelBlocks['output_pixels'][1] = k2.astype(props['pixelType'], copy=False)
            pixelBlocks['output_pixels'][2] = k3.astype(props['pixelType'], copy=False)
            pixelBlocks['output_pixels'][3] = k4.astype(props['pixelType'], copy=False)

            return pixelBlocks

    def updateKeyMetadata(self, names, bandIndex, **keyMetadata):
        if bandIndex == -1:                             # dataset-level properties
            keyMetadata['datatype'] = 'Processed'       # outgoing dataset is now 'Processed'
        elif bandIndex == 0:                            # properties for the first band
            keyMetadata['wavelengthmin'] = None         # reset inapplicable band-specific key metadata
            keyMetadata['wavelengthmax'] = None
            keyMetadata['bandname'] = 'KMeans'
        return keyMetadata

# ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ## ----- ##
