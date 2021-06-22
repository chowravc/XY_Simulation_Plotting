import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import glob
from PIL import Image
import shutil
import os

def making_mask():
	
	img = Image.open('./mask/polariser_analyser.png')

	numpydata = np.asarray(img)

	trans = np.transpose(numpydata, (2, 0, 1))

	img = Image.fromarray(trans[0], 'L')
	img.save('./mask/polariser_analyser_mask.png')
	# img.show()

	print('\nExported Mask.')

# Main bulk of code
def main(inDir, name):

	# IMPORTANT PUT EXPECTED NUMBER OF + AND - DEFECTS
	pdNum = 2
	ndNum = 2

	# Saving zoom in on defects
	pad = 20

	# Clear plt
	plt.clf()

	# Reduction factor for vector field
	reduction = 20

	## Reading files

	# Reading director angles
	directorMat = np.loadtxt('input/' + inDir + '/director_' + name + '.dat')

	# Reading background image
	im = Image.open('input/' + inDir + '/' + name + '.tiff')

	# Reading defect locations
	defectMat = np.loadtxt('input/' + inDir + '/label_' + name + '.dat')

	## Storing values for vector plot BIG PLOT
	X = []
	Y = []
	U = []
	V = []
	colorAngles = []

	# Compile values
	for i, line in enumerate(directorMat):

		if i%reduction == 0:

			for j, angle in enumerate(line):

				if j%reduction == 0:

					X.append(j)
					Y.append(i)

					U.append(np.cos(angle))
					V.append(np.sin(angle))

					colorAngles.append(angle)

	# Convert to numpy arrays
	X = np.asarray(X)
	Y = np.asarray(Y)
	U = np.asarray(U)
	V = np.asarray(V)
	colorAngles = np.asarray(colorAngles)/(2*np.pi)

	# Set colors
	colormap = cm.viridis

	## Find the locations of defects
	plusLocs = np.where(defectMat == 1.0)
	plusX = plusLocs[1]
	plusY = plusLocs[0]

	minusLocs = np.where(defectMat == -1.0)
	minusX = minusLocs[1]
	minusY = minusLocs[0]

	## Plot results
	plt.title(name)

	# Plot image
	plt.imshow(im)

	# Plot vector field
	vField = plt.quiver(X, Y, U, V, color=colormap(colorAngles), edgecolor='k')
	cbar = plt.colorbar(vField, label='Angle (Revolutions)')

	cbarlims = np.linspace(0, 1, 11)
	cbar.set_ticks(cbarlims)

	# Plot defects
	plt.scatter(plusX, plusY, color='r', s=4)
	plt.scatter(minusX, minusY, color='b', s=4)

	# Add legend for defect polarity
	red_patch = mpatches.Patch(color='red', label='Plus: '+str(len(plusX))) # positive
	blue_patch = mpatches.Patch(color='blue', label='Minus: '+str(len(minusX))) # negative
	black_patch = mpatches.Patch(color='black', alpha=0, label='Total: '+str(len(plusX)+len(minusX))) # total

	# Plot legend
	plt.legend(title="Defects", handles=[red_patch, blue_patch, black_patch], loc='center left', fancybox=True, bbox_to_anchor=(-0.3, 0))

	# temp storage
	tempStore = 'output/temp.tiff'
	if len(glob.glob(tempStore)) != 0:
		os.remove(tempStore)

	# Output plot image
	outPlot = 'output/' + inDir + '/' + name + '.tiff'

	# Make output directory
	if len(glob.glob('output/')) == 0:
		os.mkdir('output/')

	# Make outdir
	if len(glob.glob('output/'+inDir+'/')) != 0:
		pass
		# shutil.rmtree('output/'+inDir+'/')
	else:
		os.mkdir('output/'+inDir+'/')

	# Save figure. MAKE SURE THE EXPECTED NUMBER OF DEFECTS IS HERE. IMPORTANT!
	if len(plusX) == pdNum and len(minusX) == ndNum:

		plt.savefig(tempStore, dpi=200)

		# Create plus defect outdir
		if len(glob.glob('output/' + inDir + '/plus/')) == 0:
			os.mkdir('output/' + inDir + '/plus/')

		# Create minus defect outdir
		if len(glob.glob('output/' + inDir + '/minus/')) == 0:
			os.mkdir('output/' + inDir + '/minus/')

		# Save plus defect zooms
		for i in range(len(plusX)):

			zoomName = 'output/' + inDir + '/plus/' + name + '_d_' + str(plusX[i]) + '_' + str(plusY[i]) + '.tiff'

			left = plusX[i] - pad
			right = plusX[i] + pad

			up = plusY[i] - pad
			down = plusY[i] + pad

			plt.clf()

			## Plot results
			plt.title(name+'_d_'+str(plusX[i])+'_'+str(plusY[i]))

			# Plot image
			plt.imshow(im)

			# Zoom in
			plt.xlim(left, right)
			plt.ylim(up, down)

			# Save figure
			plt.savefig(zoomName)

		for i in range(len(minusX)):

			zoomName = 'output/' + inDir + '/minus/' + name + '_d_' + str(minusX[i]) + '_' + str(minusY[i]) + '.tiff'

			left = minusX[i] - pad
			right = minusX[i] + pad

			up = minusY[i] - pad
			down = minusY[i] + pad

			plt.clf()

			## Plot results
			plt.title(name+'_d_'+str(plusX[i])+'_'+str(plusY[i]))

			# Plot image
			plt.imshow(im)

			# Zoom in
			plt.xlim(left, right)
			plt.ylim(up, down)

			# Save figure
			plt.savefig(zoomName)

		## Add polariser + analyser
		# Reading mask
		mask = Image.open('./mask/polariser_analyser_mask.png')

		# Divide by max pixel brightness
		maskArray = np.asarray(mask)//255

		# Read image again
		image = Image.open(tempStore)

		# Convert to numpy array and ta
		imArray = np.transpose(np.asarray(image), (2, 0, 1))

		# Storing output of image
		output = []

		# Multiply each channel by mask
		for channel in imArray:

			output.append(channel*maskArray)

		# Rearrage tensor
		output = np.transpose(np.asarray(output[:3]), (1, 2, 0))

		# Convert numpy array to image
		outIm = Image.fromarray(output, 'RGB')

		# Save the processed image
		outIm.save(outPlot)

	# Show figure
	# plt.show()

	# Clear plt
	plt.clf()

if __name__ == '__main__':

	making_mask()

	# File name
	inputDir = 'Oblique2deg_Decross30deg_2Defects'
	ns = glob.glob('input/'+inputDir+'/*.tiff')

	for i, n in enumerate(ns):
		print(str(100*i/len(ns))[:4]+'%')
		main(inputDir, n.split('\\')[-1].split('.')[0])