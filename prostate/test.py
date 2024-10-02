import numpy as np
import matplotlib.pyplot as plt
import h5py

def load_ims(fname):
    '''
    This function loads the .ims imaris image of the leica confocal microscope.
    '''
    # load .ims file into an h5py group
    f = h5py.File(fname, 'r')
    
    # Unpacking data into 3d images (one per channel)
    im_cancer = np.array(f['DataSet/ResolutionLevel 0/TimePoint 0/Channel 0'].get('Data'))
    im_cyto = np.array(f['DataSet/ResolutionLevel 0/TimePoint 0/Channel 1'].get('Data'))
    im_bf = np.array(f['DataSet/ResolutionLevel 0/TimePoint 0/Channel 2'].get('Data'))
    im_nuclei = np.array(f['DataSet/ResolutionLevel 0/TimePoint 0/Channel 3'].get('Data'))
    
    # Repacks data into a 4d image
    im4d = np.array([im_cancer, im_cyto, im_nuclei, im_bf])
    
    return im4d

# loading the leica data
im4d = load_ims('data/leica_prostate_cell_scan.ims')

# Sum along the z-axis to generate 2D projections
imr_proj = im4d[0, :,:,:].sum(axis=0)
img_proj = im4d[1, :,:,:].sum(axis=0)
imb_proj = im4d[2, :,:,:].sum(axis=0)
imy_proj = im4d[3, :,:,:].sum(axis=0)

# Stack the 3 channels (R, G, B) for the projection image
im_proj = np.stack([imr_proj, img_proj, imb_proj], axis=2)

# Choose a specific z-slice
zslice = 8
imr_slice = im4d[0, zslice,:,:]
img_slice = im4d[1, zslice,:,:]
imb_slice = im4d[2, zslice,:,:]
imy_slice = im4d[3, zslice,:,:]

# Stack the 3 channels (R, G, B) for the z-slice image
im_slice = np.stack([imr_slice, img_slice, imb_slice], axis=2)

# Normalize both images
im_proj_normalized = im_proj / im_proj.max()
im_slice_normalized = im_slice / im_slice.max()

# Create a figure with 2 subplots side by side
plt.figure(figsize=[15,10])

# Display the projection image on the left
plt.subplot(1, 2, 1)
plt.imshow(im_proj_normalized)
plt.axis('off')
plt.title('Projection Image')

# Display the z-slice image on the right
plt.subplot(1, 2, 2)
plt.imshow(im_slice_normalized)
plt.axis('off')
plt.title(f'Z-slice {zslice}')

# Show the figure
plt.show()