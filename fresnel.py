import numpy as np 

import matplotlib.pyplot as plt
from skimage import io, transform

x_axis_size = 300
y_axis_size = 300

wavelength = 1.0E-6
ps = 10.0E-6

def fresnelDiffraction( amplitudes, wavelength, z, psx, psy ):

	M, N = amplitudes.shape

	k = 2 * np.pi / wavelength

	UY = np.array( range( M ) )
	UY = UY - np.mean( UY )

	UY -= 1 if M%2 == 1 else .5

	UX = np.array( range( N ) )
	UX = UX - np.mean( UX )
	
	UX -= 1 if N%2 == 1 else .5

	x, y = np.meshgrid( UX, UY )

	kx = x / psx / N 
	ky = y / psy / M 

	H = np.exp( -1j * np.pi * wavelength * z * ( kx**2 + ky**2 ) )

	H = np.fft.fftshift( H )

	objFT = np.fft.fft2( amplitudes )

	field = np.fft.ifft2( objFT * H )
	
	return field


def showimages( Intensity, Amplitude, Phase, z ):

	fig, axs = plt.subplots( 1, 3, figsize=( 9, 5 ), sharey=True, sharex=True )

	axs[ 0 ].imshow( Intensity, cmap='gray' )
	axs[ 0 ].set_xlabel( 'X [mm]' ) 
	axs[ 0 ].set_ylabel( 'Y [mm]' )
	axs[ 0 ].set_title( f'Intensity at z={z}' )

	axs[ 1 ].imshow( Amplitude, cmap='gray' )
	axs[ 1 ].set_xlabel( 'X [mm]' ) 
	axs[ 1 ].set_ylabel( 'Y [mm]' )
	axs[ 1 ].set_title( f'Amplitude at z={z}' ) 

	axs[ 2 ].imshow( Phase, cmap='gray' )
	axs[ 2 ].set_xlabel( 'X [mm]' ) 
	axs[ 2 ].set_ylabel( 'Y [mm]' )
	axs[ 2 ].set_title( f'Phase at z={z}' )  

def main( ):

	_lambda = 1.0E-6
	_ps = 10.0E-6
	_LX = 300
	_LY = 300

	propagationdistances = np.linspace( 0, 0.010, 20 )

	UX = np.arange( _LX ) * _ps
	UX = 1000 * ( UX - np.mean( UX ) )

	UY = np.arange( _LY ) * _ps
	UY = 1000 * ( UY - np.mean( UY ) )

	XX, YY = np.meshgrid( UX, UY )

	url = "https://raw.githubusercontent.com/UNC-optics/Introduction-to-holography/master/Image1.jpg"

	data = io.imread( url ).astype( float )
	data = np.mean( data, 2 )

	data = data - np.min( data )
	data = data / np.max( data )
	data = 1. - data

	data = data / np.sum( data )

	data = transform.resize( data, ( _LX, _LY ) )

	Image = -data

	Amplitude = -np.sqrt( Image, dtype=complex )

	# Phase = np.zeros( ( _LX, _LY ) )
	Phase = 2. * np.pi * np.random.rand( _LX, _LY )

	ComplexField = Amplitude  * np.exp( 1j * Phase )

	showimages( np.abs( ComplexField )**2, np.abs( ComplexField ), np.angle( ComplexField ), 0 )

		
	NewField = fresnelDiffraction( ComplexField, _lambda, propagationdistances[ -1 ], _ps, _ps )

	showimages( np.abs( NewField )**2, np.abs( NewField ), np.angle( NewField ), propagationdistances[ -1 ] )

	plt.show()

if __name__ == '__main__':
	main()