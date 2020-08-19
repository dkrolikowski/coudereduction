##### FUNCTIONS FOR REDUCING COUDE ECHELLE SPECTRA #####
##### Daniel Krolikowski, Aaron Rizzuto #####
##### Used on data from the Tull coude spectrograph on the 2.7m at McDonald #####

###################################################################################################

##### IMPORTS #####

from __future__ import print_function

import glob, os, pdb, pickle, mpyfit, cosmics

import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.time import Time
from scipy import signal
from continuumfit import getContinuum

###################################################################################################

###### FUNCTIONS ######

## Function: Make the header information file ##
def Header_Info( Conf ):

    files   = glob.glob( '*.fits' )

    outfile = open( Conf.InfoFile, 'wb' )
    
    heading = 'File,Object,RA,DEC,Type,ExpTime,Order,Airmass,UTdate,UT,gain,rdn,zenith\n'
    outfile.write( heading )
    
    for f in range( len( files ) ):
        head  = fits.open(files[f])[0].header
        
        itype = head["imagetyp"]
        if len(files[f].split('.')) > 2: itype = "unknown"
        
        ra = ''
        dec = ''
        if "RA"  in head.keys(): ra  = head["RA"]
        if "DEC" in head.keys(): dec = head["DEC"]

        air = '100'
        if "airmass" in head.keys(): air = str(head["airmass"])
        
        exptime = str(head["exptime"])
        
        if "GAIN3" in head.keys():
            gain = str(head["gain3"])
            rdn  = str(head["rdnoise3"])
        else:
            gain = str(head["gain2"])
            rdn  = str(head["rdnoise2"])

        line = ','.join( [ files[f], head['object'], ra, dec, itype, exptime, head['order'], air, head['DATE-OBS'], head['UT'], gain, rdn, head['ZD'] ] )
        outfile.write( line + '\n' )

        outfile.flush()

    outfile.close()
    
    return None

#############################################

### Section: Basic calibration functions (bias, flat, bad pixel mask) ###

## Function: Make the master bias frame ##
def Build_Bias( files, readnoise ):

    testdata = fits.open( files[0] )[0].data
    biascube = np.zeros( ( len( files ), testdata.shape[0], testdata.shape[1] ) )

    for f in range( len( files ) ):
        biascube[f] = fits.open( files[f] )[0].data
        
    SuperBias = {}
    
    SuperBias['vals'] = np.nanmedian( biascube, axis = 0 )
    SuperBias['errs'] = np.sqrt( SuperBias['vals'] + readnoise ** 2.0 )

    return SuperBias

## Function: Make the master flat frame ##
def Build_Flat_Field( files, readnoise, SuperBias ):

    testdata = fits.open( files[0] )[0].data
    flatcube = np.zeros( ( len( files ), testdata.shape[0], testdata.shape[1] ) )

    for f in range( len( files ) ):
        flatcube[f] = fits.open( files[f] )[0].data

    FlatField = {}
    
    flatmedian = np.nanmedian( flatcube, axis = 0 )
    flatvalues = flatmedian - SuperBias['vals']     # Subtract off the bias from the flat!
    
    FlatField['errs'] = np.sqrt( flatmedian + readnoise ** 2.0 + SuperBias['errs'] ** 2.0 ) # Photon noise and bias error propagation
    FlatField['vals'] = flatvalues - flatvalues.min()
    FlatField['errs'] = FlatField['errs'] / FlatField['vals'].max()
    FlatField['vals'] = FlatField['vals'] / FlatField['vals'].max()

    return FlatField

## Function: Make the bad pixel mask ##
def Make_BPM( SuperBias, FlatField, Conf ):

    cutbias = np.percentile( SuperBias, Conf.BPMlimit )
    BPM     = np.where( ( SuperBias > cutbias ) | ( FlatField <= 0.0001 ) )
    
    return BPM

## Function: Call sub-functions and return bias, flat, BPM ##
def Basic_Cals( BiasInds, FlatInds, FileInfo, Conf ):
    
    if Conf.doCals == True: # If we need to generate calibration files
        # Create master bias
        if Conf.verbose: print( 'Reading Bias Files' )
        biasrdn   = FileInfo.rdn[BiasInds].values / FileInfo.gain[BiasInds].values
        SuperBias = Build_Bias( FileInfo.File[BiasInds].values, biasrdn[0] )
        pickle.dump( SuperBias, open( Conf.rdir + 'bias.pkl', 'wb' ) )
        
        # Create master flat
        if Conf.verbose: print( 'Reading Flat Files' )
        flatrdn   = FileInfo.rdn[FlatInds].values / FileInfo.gain[FlatInds].values
        FlatField = Build_Flat_Field( FileInfo.File[FlatInds].values, flatrdn[0], SuperBias )
        pickle.dump( FlatField, open( Conf.rdir + 'flat.pkl', 'wb' ) )
        
        # Create the bad pixel mask
        if Conf.verbose: print( 'Creating the bad pixel mask' )
        BPM = Make_BPM( SuperBias['vals'], FlatField['vals'], Conf )
        pickle.dump( BPM, open( Conf.rdir + 'bpm.pkl', 'wb' ) )
        
    elif Conf.doCals == False: # If we're reading in already generated cal files
        if Conf.verbose: print( 'Reading in premade Bias, Flat, and BPM files' )
        SuperBias = pickle.load( open( Conf.rdir + 'bias.pkl' ) )
        FlatField = pickle.load( open( Conf.rdir + 'flat.pkl' ) )
        BPM = pickle.load( open( Conf.rdir + 'bpm.pkl' ) )
        
    plt.clf() # Plot the bias
    plt.imshow( np.log10( SuperBias['vals'] ), cmap = plt.get_cmap( 'gray' ), aspect = 'auto', interpolation = 'none' )
    plt.colorbar(); plt.title( str(np.nanmedian(SuperBias['vals'])) ); plt.savefig( Conf.rdir + 'plots/bias.pdf' ); plt.clf()
    
    plt.clf() # Plot the flat
    plt.imshow( np.log10( FlatField['vals'] ), cmap = plt.get_cmap( 'gray' ), aspect = 'auto', interpolation = 'none' )
    plt.colorbar(); plt.savefig( Conf.rdir + 'plots/flat.pdf' ); plt.clf()
    
    plt.clf() # Plot the BPM
    plt.imshow( np.log10( SuperBias['vals'] ), aspect = 'auto', interpolation = 'none' )
    plt.plot( BPM[1], BPM[0], 'r.', ms = 1 ) # Invert x,y for imshow
    plt.savefig( Conf.rdir + 'plots/bpm.pdf' ); plt.clf()

    return SuperBias, FlatField, BPM
    
#############################################
    
### Section: Generate data cubes -- 2D spectral cubes for arc and object exposures ###

## Function: Make the data cubes! With cosmic subtraction, error propagation ##
def Make_Cube( Files, ReadNoise, Gain, DarkVal, Conf, Bias = None, Flat = None, BPM = None, cossub = False ):

    for f in range( len( Files ) ): # Loop through all files
        frame = fits.open( Files[f] )[0].data

        if f == 0:
            Cube = np.zeros( ( len( Files ), frame.shape[0], frame.shape[1] ) )
            SNR  = np.zeros( ( len( Files ), frame.shape[0], frame.shape[1] ) )
                        
        Cube[f] = frame - DarkVal[0] # Cube value, subtract dark is there is one
        CubeErr = np.sqrt( Cube[f] + DarkVal[0] + ReadNoise[f] ** 2.0 ) # Noise
        
        # Perform cosmic subtraction, if specified
        if cossub:
            cos = cosmics.cosmicsimage( Cube[f], gain = Gain[f], readnoise = ReadNoise[f], sigclip = 5.0, sigfrac = 0.3, objlim = 5.0 )
            cos.run( maxiter = Conf.cos_iters )
            Cube[f] = cos.cleanarray
                        
        CBerrVal = CubeErr ** 2.0 / Cube[f] ** 2.0
        FerrVal  = 0.0
        
        if Bias is not None: # If we're subtracting off the bias
            Cube[f] -= Bias['vals']
            CBerrVal = ( CubeErr ** 2.0 + Bias['errs'] ** 2.0 ) / Cube[f] ** 2.0
        if Flat is not None: # If we're dividing out the flat
            Cube[f] /= Flat['vals']
            FerrVal  = ( Flat['errs'] / Flat['vals'] ) ** 2.0
            
        FullErr = np.sqrt( Cube[f] ** 2.0 * ( CBerrVal + FerrVal ) ) # Total Error
        
        SNR[f]  = Cube[f] / FullErr # Compute the signal to noise cube
                    
        if BPM is not None: # If there's a bad pixel
            Cube[f,BPM[0],BPM[1]] = np.nanmedian( Cube[f] )  # Is setting a bad pixel to the median right?
            SNR[f,BPM[0],BPM[1]]  = 1e-4 # Well I'm setting the S/N to effectively 0 so...

        wherenans = np.where( np.isnan( Cube[f] ) )
        Cube[f,wherenans[0],wherenans[1]] = np.nanmedian( Cube[f] ) # I might as well just do the same as for the BPM
        SNR[f,wherenans[0],wherenans[1]]  = 1e-4 # Set a nan pixel to have an effectively 0 S/N
        
    return Cube, SNR

## Function: Call Make_Cube to return arc and object cubes
def Return_Cubes( ArcInds, ObjInds, FileInfo, DarkCube, Bias, Flat, BPM, Conf ):

    if Conf.verbose: print( 'Generating arc spectral cubes' )
    ReadNoise       = FileInfo.rdn[ArcInds] / FileInfo.gain[ArcInds]
    DarkVal         = DarkCube[ArcInds] / FileInfo.gain[ArcInds]
    GainVals        = FileInfo.gain[ArcInds].values
    ArcCube, ArcSNR = Make_Cube( FileInfo.File[ArcInds].values, ReadNoise.values, GainVals, DarkVal.values, Conf, Bias = Bias )

    if Conf.verbose: print( 'Generating object spectral cubes' )
    ReadNoise       = FileInfo.rdn[ObjInds] / FileInfo.gain[ObjInds]
    DarkVal         = DarkCube[ObjInds] / FileInfo.gain[ObjInds]
    GainVals        = FileInfo.gain[ObjInds].values
    ObjCube, ObjSNR = Make_Cube( FileInfo.File[ObjInds].values, ReadNoise.values, GainVals, DarkVal.values, Conf, Bias = Bias, Flat = Flat, BPM = BPM, cossub = Conf.CosmicSub )

    return ArcCube, ArcSNR, ObjCube, ObjSNR

#############################################

### Section: The trace! ###

## Function: Get peak values of trace for a slice of the flat ##
def Start_Trace( flatslice, percent ):

    fgrad    = np.gradient( flatslice )
    cutvalue = np.nanpercentile( abs( fgrad ), percent )

    orderzeros = []
    last       = 0

    # Find peaks based on the gradient of the flat slice
    for i in range( 6, flatslice.shape[0] ):
        if fgrad[i] > cutvalue or last == 0:
            if 100 > i - last > 20 or last == 0:
                orderzeros.append( i + 11 )
                last = i

    orderzeros = np.array( orderzeros )

    for i in range( len( orderzeros ) ): # Go through and recenter the peaks found
        o      = orderzeros[i]
        cutoff = flatslice[o] * ( 0.7 )
        left   = o - 15 + np.where( flatslice[o-15:o] <= cutoff )[-1]
        right  = o + np.where( flatslice[o:o+20] <= cutoff )[-1]
        if len( left ) == 0 or len( right ) == 0:
            orderzeros[i] = o
        else:
            orderzeros[i] = ( right[0] + left[-1] ) / 2

    ordervals = flatslice[orderzeros]
    
    return orderzeros, ordervals

## Function: Use flat slices to find starting values for the trace ##
def Find_Orders( Flat, orderstart ):
    # Uses flat slices at edge and in middle and uses that to refine initial points

    midpoint = ( ( Flat.shape[1] + orderstart ) / 2 ) + 100

    startzeros, startvals = Start_Trace( Flat[:,orderstart], 60.0 ) # Get peaks for edge of flat
    midzeros, midvals     = Start_Trace( Flat[:,midpoint], 45.0 )   # Get peaks for middle of flat
    
#    plt.clf()
#    plt.figure()
#    plt.plot( Flat[:,orderstart], 'k-' ); plt.plot( startzeros, startvals, 'r+' )
#    plt.figure()
#    plt.plot( Flat[:,midpoint], 'b-' ); plt.plot( midzeros, midvals, 'm+' )
#    plt.show()
        
    # By hand remove extra orders that are present at the midpoint
    midzeros = midzeros[2:]
    midvals  = midvals[2:]

    # Calculate a slope between the two to refine and smooth out trace starting points across orders
    slopes = []
    dx     = Flat.shape[1] + orderstart - midpoint
    
    for i in range( 5, 50 ):
        dy = float( startzeros[i] - midzeros[i] )
        slopes.append( dy / dx )

    slopefit = np.polyfit( range( 5, 50 ), slopes, 2 )
    
    finalzeros = np.round( midzeros + np.polyval( slopefit, range( len( midzeros ) ) ) * dx ).astype( int )
    finalvals  = Flat[finalzeros, orderstart]
            
    return finalzeros, finalvals

## Function: Use order starting points to calculate full trace from bright object spectra ##
def Full_Trace( brightcube, orderzeros, orderstart ):
    # Get the full 2D trace using the brightest object images

    numord = len( orderzeros )
    trace  = np.zeros( ( brightcube.shape[0], numord, brightcube.shape[2] + orderstart ) )

    for f in range( brightcube.shape[0] ):
        for pix in range( 1, brightcube.shape[2] + orderstart + 1 ):
            prev = orderzeros
            if pix > 1: prev = trace[f,:,-pix+1]
            m1d = brightcube[f,:,-pix+orderstart]
            for o in range( numord ):
                edge1 = int(prev[o] - 3)
                if edge1 < 0: edge1 = 0
                trace[f,o,-pix] = edge1 + np.argmax( m1d[edge1:edge1+6] )
                if pix != 1:
                    if np.abs( trace[f,o,-pix] - prev[o] ) > 2:
                        trace[f,o,-pix] = prev[o]

    return trace

## Function: Fit the full trace and correct outlier bad orders ##
def Fit_Trace( Trace ):
    # Fit the trace with a 2nd order polynomial (the cubic term in the 3rd order fit was basically 0 for all orders)
    # Also go through and fit the linear and quadratic terms as function of order number -- fix bad orders!

    FitTrace = np.zeros( ( Trace.shape[0], 2048 ) )
    fitpars  = np.zeros( ( Trace.shape[0], 3 ) )
    
    for o in range( Trace.shape[0] ): # Do initial fit for the trace along each order
        
        rng_tofit   = range( 512, Trace.shape[1] ) # Only fit the latter 3/4 of the order, first 1/4 can be bad due to counts, etc.
        
        poly        = np.polyfit( np.array( rng_tofit, dtype = float ), Trace[o,rng_tofit], 2 )
        fitpars[o]  = poly
        
    for i in [ 0, 1 ]: # Redetermine linear and quadratic terms in the fits (leave zero point alone)
        
        hyperpars = np.polyfit( np.arange( fitpars.shape[0] ), fitpars[:,i], 2 )
        hyperfit  = np.polyval( hyperpars, np.arange( fitpars.shape[0] ) )
                
        meddiff   = np.median( np.abs( fitpars[:,i] - hyperfit ) )
        mask      = np.where( np.abs( fitpars[:,i] - hyperfit ) <= 5 * meddiff )[0] # Correct orders more than 5 "sigma" bad

        hyperpars = np.polyfit( mask, fitpars[mask,i], 3 )
        hyperfit  = np.polyval( hyperpars, np.arange( fitpars.shape[0] ) )
        
        fitpars[:,i] = hyperfit
        
    for o in range( Trace.shape[0] ): # Calculate fitted trace from corrected/final polynomial fits
        
        FitTrace[o] = np.polyval( fitpars[o], np.arange( 2048 ) )

    return FitTrace

## Function: Call above functions to get initial trace, calculate full trace, and return fitted trace ##
def Get_Trace( Flat, Cube, Conf ):

    if Conf.doTrace == True: # If we need to calculate the trace
        
        orderstart            = -33 # Set where the orders are starting (in this case, pixel 2047 not pixel 2048)
        orderzeros, ordervals = Find_Orders( Flat, orderstart ) # Find initial values for trace
        
        if Conf.verbose: print( 'Performing preliminary trace' )

        # Plot the preliminary trace
        
#        plt.clf()
#        plt.figure()
#        plt.plot( np.arange( orderzeros.size ), orderzeros, 'r+' )
#
#        plt.figure()
#        plt.plot( np.diff( orderzeros ), 'r+' )
#        
        plt.figure()
        plt.plot( Flat[:,orderstart], 'k-' ); plt.plot( orderzeros, ordervals, 'r+' )
        plt.savefig( Conf.rdir + 'plots/prelimtrace.pdf' ); plt.clf()
#        plt.show()

        # Determine the brightest object frames for trace finding
        meds      = [ np.nanmedian( Cube[i,:,:2048] ) for i in range( Cube.shape[0] ) ]
        abovemedi = np.where( meds >= np.nanpercentile( meds, Conf.MedCut ) )
        abovemed  = Cube[abovemedi]

        trace     = Full_Trace( abovemed, orderzeros, orderstart ) # Get full trace
        MedTrace  = np.median( trace, axis = 0 ) # Get median trace
        FitTrace  = Fit_Trace( MedTrace ) # Fit the median trace
        
        # Make sure the top order is a full order and doesn't spill over top of image
        if FitTrace[0,-1] <= 10.0: # 10 pixels is rough width of an order, be a bit conservative
            MedTrace = MedTrace[1:]
            FitTrace = FitTrace[1:]

        if Conf.verbose: print( 'Saving median and fitted trace to file' )
        pickle.dump( MedTrace, open( Conf.rdir + 'median_trace.pkl', 'wb' ) )
        pickle.dump( FitTrace, open( Conf.rdir + 'fitted_trace.pkl', 'wb' ) )
        
        # Plot the full trace
        plt.clf()
        plt.imshow( np.log10( Flat ), aspect = 'auto', cmap = plt.get_cmap( 'gray' ) )
        for o in range( MedTrace.shape[0] ): plt.plot( MedTrace[o], 'r', lw = 1.0 )
        for o in range( FitTrace.shape[0] ): plt.plot( FitTrace[o], 'b', lw = 1.0 )
        plt.xlim( 0, 2048 ); plt.ylim( 2048, 0 ); plt.savefig( Conf.rdir + 'plots/trace.pdf' ); plt.clf()

    elif Conf.doTrace == False: # If trace has already been calculated
        
        if Conf.verbose: print( 'Reading in premade Trace and plotting on Flat:' )
        MedTrace = pickle.load( open( Conf.rdir + 'median_trace.pkl', 'rb' ) )
        FitTrace = pickle.load( open( Conf.rdir + 'fitted_trace.pkl', 'rb' ) )

    return MedTrace, FitTrace

#############################################

### Section: Extracting 1D spectra from the 2D spectral cubes ###

## Function: Calculate difference between data and model for mpyfit ## 
def Least( p, args ):
    
    X, vals, err, func = args # Unpack arguments
    
    if err is not None: # If there is an error array provided
        dif = ( vals - func( X, p ) ) / err
    else:
        dif = vals - func( X, p )

    return dif.ravel() # Use ravel() to turn multidimensional arrays to 1D

## Function: 2D model of an order (each pixel is a gaussian with variation along dispersion direction) ##
def OrderModel( X, p, return_full = False ):
    
    x, y = X
    
    means  = p[2] * x ** 2.0 + p[1] * x + p[0] # Trace of the order -- parabola
    peaks  = p[5] * x ** 2.0 + p[4] * x + p[3] # Peak shape curve -- parabola
    sigmas = p[9] * x ** 3.0 + p[8] * x ** 2.0 + p[7] * x + p[6] # Sigma curve -- cubic
    
    # Full model
    model  = peaks * np.exp( - ( y - means ) ** 2.0 / ( 2.0 * sigmas ** 2.0 ) ) + p[10]
    
    if return_full == False: return model    # If we just want the model
    else: return model, means, peaks, sigmas # If we need all of the model constituents

## Function: Simple 1D Gaussian model ##
def GaussModel( X, p ):
    
    x = X

    model = p[0] * np.exp( - ( x - p[1] ) ** 2.0 / ( 2.0 * p[2] ** 2.0 ) ) + p[3]

    return model

## Function: Extraction! All wrapped into one function ##
def Extractor( Cube, SNR, Trace, Conf, quick = True, arc = False, nosub = True ):

    # Initialize arrays 
    flux  = np.zeros( ( Cube.shape[0], Trace.shape[0], Trace.shape[1] ) )
    error = flux * 0.0

    for frm in range( Cube.shape[0] ):# Loop through all of the frames
        if Conf.verbose: print( 'Extracting Frame', str( frm + 1 ), 'out of', str( Cube.shape[0] ) )
        
        thisfrm = Cube[frm,:,:]
        thissnr = SNR[frm,:,:]
        
        for ord in range( Trace.shape[0] ): # Loop through all of the orders
            if Conf.verbose: print( 'Extracting Ord', str( ord + 1 ), 'out of', str( Trace.shape[0] ), 'for frame', str( frm + 1 ), 'of', str( Cube.shape[0] ) )

            # Set up rectangle around order that will be fit
            tblock = np.zeros( ( Trace.shape[1], 16 ) )
            tsnr   = tblock.copy()
            x, y   = [ c.T for c in np.meshgrid( np.arange( tblock.shape[0] ), np.arange( tblock.shape[1] ) ) ]
            
            for pix in range( Trace.shape[1] ): # Put image and snr into the block used for extraction (interpolation? A little unsure about that)
                low           = np.round(Trace[ord,pix]).astype(int) - 10
                high          = np.round(Trace[ord,pix]).astype(int) + 10
                tblock[pix,:] = np.interp( np.linspace(Trace[ord,pix] - 8, Trace[ord,pix] + 8, 16 ), np.linspace(low,high,20), thisfrm[low:high,pix] )
                tsnr[pix,:]   = np.interp( np.linspace(Trace[ord,pix] - 8, Trace[ord,pix] + 8, 16 ), np.linspace(low,high,20), thissnr[low:high,pix] )

            tsnr[np.isnan(tsnr)] = 1e-5 # Any nans have very low SNR values
                        
            if (quick == False) & (arc == False): # If we're doing the full extraction!

                # Clean obvious high outliers -- Should probably reinstate this!
#                toohigh         = np.where( tblock > 15.0 * np.median( tblock ) )
#                tblock[toohigh] = np.median( tblock )
#                tsnr[toohigh]   = 0.000001
                
                tnoise = np.absolute( tblock / ( tsnr ) ) # Get just the error block ( signal / ( signal / noise ) )

                # Clean zero values (often due to chip artifacts that aren't caught)
                
                if np.median( tblock ) <= 0.0: val2use = 0.0 # Will use median to replace values too low, but if median is somehow below 0, set them to 0
                else: val2use = np.median( tblock )
                
                toolow         = np.where( tblock <= 0.0 )
                tblock[toolow] = val2use # Set too low values to value
                tsnr[toolow]   = 1e-5    # SNR for too low values very small
                
                # Change the values in the noise block
                if val2use == 0.0: tnoise[toolow] = np.median( np.abs( tblock[tblock>0] ) )
                else: tnoise[toolow] = np.median( tblock ) * 2.0
                
                # Fit the 2D order block with the order model from above! Set initial guesses for the parameters from block values
                p0              = np.zeros( 11 )
                p0[[0,3,6,10]]  = [ np.argmax( tblock[0,:] ), np.median( tblock[:,np.argmax( tblock[1000,:] )] ), 2.0, np.percentile( tblock, 2.0 ) ]

                ordpars, ordres = mpyfit.fit( Least, p0, ( (x,y), tblock, tnoise, OrderModel ) ) # Perform fit!

                if ordres['status'] == -16:
                    print( 'mpyfit for this order failed?' )
                    pdb.set_trace()

                bestmod, besttrace, bestpeak, bestsigmas = OrderModel( (x,y), ordpars, return_full = True ) # Get model and residuals
                bestresid                                = ( tblock - bestmod ) / tnoise

                # Now remove points that are obvious outliers from the best model!
                cutresid       = np.percentile( bestresid.flatten(), 99.99 )
                badcut         = np.where( bestresid > cutresid )
                tblock[badcut] = np.median( tblock ) # Bad values set to median
                tsnr[badcut]   = 1e-5 # Bad values set to SNR very small
                
                # Save the best fit trace, peak shape, sigma shape, and background value!
                thistrace      = besttrace[:,0].copy()
                thispeak       = bestpeak[:,0].copy()
                thissigma      = bestsigmas[:,0].copy()
                block_bg       = ordpars[5]

            savesigma = np.zeros( Trace.shape[1] )
            savepeak  = np.zeros( Trace.shape[1] )
            savebg    = np.zeros( Trace.shape[1] )
            savespot  = np.zeros( Trace.shape[1] )
            
            for pix in range( Trace.shape[1] ): # Now go pixel by pixel along order! To collapse 2D to 1D

                slice          = tblock[pix,:]
                slice_snr      = tsnr[pix,:]
                qwe            = np.where( slice_snr < 0.0 )[0] # If SNR anywhere is negative for some reason... This should never be used?
                slice_snr[qwe] = 1e-5                           # Set the SNR to something very small!
                thisx          = np.arange(len(slice))
                snoise         = np.absolute(slice/slice_snr)
                snoise[snoise==0.0] = 0.00001

                if quick == False: # If we're doing the full extraction!
                    tspot     = thistrace[pix] # Get value of trace, peak, and sigma from 2D fit
                    tpeak     = thispeak[pix]
                    tsigma    = thissigma[pix]

                    # Initial parameters: peak height, peak centroid and peak width initalized from global order fit above, 
                    p0        = np.array( [ tpeak, tspot, tsigma, block_bg ] )
                    parinfo   = [ { 'limits': ( None, None ) } for i in range( len(p0) ) ] # Set some limits on the fit values
                    parinfo[1]['limits'] = ( p0[1] - 2.0, p0[1] + 2.0 )
                    parinfo[2]['limits'] = ( p0[2] - 2.0, p0[2] + 2.0 )

                    slicepars, sliceres  = mpyfit.fit( Least, p0, ( thisx, slice, snoise, GaussModel ), parinfo = parinfo, maxiter = 10 ) # Do 1D Gauss fit
                    
                    savesigma[pix] = slicepars[2]
                    savespot[pix]  = slicepars[1]
                    savepeak[pix]  = slicepars[0]
                    savebg[pix]    = slicepars[3]
                    bg             = slicepars[3]

                    themod = GaussModel( thisx, slicepars ) # Get best fit model and residuals
                    resids = ( slice - themod ) / snoise

                    # Detect outlier where residuals are really large
                    bads   = np.where( np.absolute( resids ) > 50 )[0] # Maybe check in on that residual value some time?
                    slice_snr[bads] = 1e-5                             # Set the SNR for those bad points very small
                    
                    if len(bads) > 3: slice_snr = slice_snr * 0.0 + 1e-5 # If there are more than 3 bad points, downweight pixel slice. Maybe look into this?

                # If we're doing quick extraction, simple background subtraction
                if quick   == True: bg = slice * 0.0 + np.min( np.absolute( slice ) ) # If quick subtraction, set background to minimum of slice
                if nosub   == True: bg = slice * 0.0 # If no subtraction set, set background to 0
                cslice     = slice - bg # Subtract background, even if set to 0

                # Sum along slice, weighted by SNR!
                flux[frm,ord,pix] = np.sum( slice_snr * cslice ) / np.sum( slice_snr ) # Make sure this is right? I think it is

                # For arc frames just do a simple sum along slice!
                if arc == True: flux[frm,ord,pix] = np.sum( cslice )
                
                # Get the final uncertainty!
                final_sig = np.sqrt( np.sum( slice_snr ** 2.0 * snoise ** 2.0 ) / np.sum( slice_snr ) ** 2.0 ) # Error
                final_snr = np.absolute( flux[frm,ord,pix] / final_sig ) # SNR

                if final_snr < 0: # Negative SNR?
                    print( 'something has gone wrong, negative errors!!??!!' )
                    pdb.set_trace()            
    
                error[frm,ord,pix] = final_sig * 1.0
    
                # Further clean bad points? Sure. Check if this is ever used?
                badlow = np.where( flux[frm,ord] <= 1 ) # Seems arbitrary?
                error[frm,ord,badlow] = np.median( flux[frm,ord] )

    return flux, error

#############################################

### Section: Wavelength calibration! From Arc to Object ###

## Function: Plotting function, does the full spectrum and zoomed-in windows ##    
def Plot_Order_Wavsol( wavsol, wavkept, spec, THAR, path, frame, order ):
    
    # First do the plot of the full order spectrum!
    
    plt.clf()
    plt.plot( wavsol, spec, 'k-', lw = 1 )
    plt.plot( THAR['wav'], THAR['logspec'], 'r-', lw = 1 )
    plt.xlim( wavsol[0] - 5.0, wavsol[-1] + 5.0 )
    for peak in wavkept:
        plt.axvline( x = peak, color = 'b', ls = ':', lw = 1 )
    plt.savefig( path + '/fullspec_ord_' + str(order) + '.pdf' ); plt.clf()
    
    # Now do the spectrum window plots, for zoom in!

    numfigs = np.ceil( np.ceil( np.ptp( wavsol ) / 10.0 ) / 6.0 ).astype(int)
    start   = np.min( wavsol )
    if numfigs > 4:
        return None
    else:
        for i in range(numfigs):
            fig   = plt.figure()
            j = 1
            while start <= np.max( wavsol ) and j < 7:
                
                window_min = start
                window_max = start + 12.0
                
                data_loc = np.where( ( wavsol >= window_min ) & ( wavsol <= window_max ) )[0]
                ref_loc  = np.where( ( THAR['wav'] >= window_min ) & ( THAR['wav'] <= window_max ) )[0]
                y_offset = np.nanpercentile( THAR['logspec'][ref_loc], 35.0 ) - np.nanpercentile( spec[data_loc], 35.0 )
                
                sbplt = 230 + j
                fig.add_subplot(sbplt)
                plt.plot( wavsol[data_loc], spec[data_loc] + 0.4 * y_offset, 'k-', lw = 1 )
                plt.plot( THAR['wav'][ref_loc], THAR['logspec'][ref_loc], 'r-', lw = 1 )
                for peak in wavkept:
                    plt.axvline( x = peak, color = 'b', ls = ':', lw = 1 )
                plt.xlim( start, start + 12.0 )
                plt.yticks( [], [] )
                low = np.round( start, 1 )
                plt.xticks( np.linspace( low, low + 12.0, 3 ), fontsize = 5 )
                start += 10.0
                j += 1
            plt.suptitle( 'Frame: ' + str(frame) + ', Order: ' + str(order) + ', Window: ' + str(i) )
            plt.savefig( path + '/specwindow_' + str(i) + '.pdf' ); plt.clf()
    
        return None

## Function: Plotting function for the wavelength solution residuals, in wav and velocity ##
def Plot_WavSol_Resids( resids, lines, cutoff, savename, tokeep = None, toreject = None ):

    wavmad = np.median( np.abs( np.abs( resids['wav'] ) - np.median( np.abs( resids['wav'] ) ) ) )
    velmad = np.median( np.abs( np.abs( resids['vel'] ) - np.median( np.abs( resids['vel'] ) ) ) )
    
    plt.clf()
    fig, (wavax, velax) = plt.subplots( 2, 1, sharex = 'all' )
    wavax.axhline( y = 0.0, color = 'k', ls = ':' )
    velax.axhline( y = 0.0, color = 'k', ls = ':' )
    if toreject is None:
        wavax.plot( lines, resids['wav'], 'ko', mfc = 'none' )
        velax.plot( lines, resids['vel'], 'ko', mfc = 'none' )
        fig.suptitle( 'Lines Used: ' + str(lines.size) + ', Cutoff: ' + str(cutoff * 0.67449) + ' $\sigma$' )
    elif tokeep is None:
        wavax.plot( lines, resids['wav'], 'kx', mfc = 'none' )
        velax.plot( lines, resids['vel'], 'kx', mfc = 'none' )
        fig.suptitle( 'Lines Used: ' + str(lines.size) + ', Cutoff: ' + str(cutoff * 0.67449) + ' $\sigma$' )
    else:
        wavax.plot( lines[tokeep], resids['wav'][tokeep], 'ko', mfc = 'none' )
        wavax.plot( lines[toreject], resids['wav'][toreject], 'kx' )
        velax.plot( lines[tokeep], resids['vel'][tokeep], 'ko', mfc = 'none' )
        velax.plot( lines[toreject], resids['vel'][toreject], 'kx' )
        fig.suptitle( 'Lines Used: ' + str(lines[tokeep].size) + ', Lines Rej: ' + str(lines[toreject].size) 
                     + ', Cutoff: ' + str(cutoff * 0.67449) +  ' $\sigma$' )
    for x in [ -cutoff, cutoff ]:
        wavax.axhline( y = x * wavmad, color = 'r', ls = '--' )
        velax.axhline( y = x * velmad, color = 'r', ls = '--' )
        
    wavax.set_ylabel( 'Resids ($\AA$)' )
    velax.set_ylabel( 'Resids (km/s)' )
    wavax.yaxis.set_label_position("right"); velax.yaxis.set_label_position("right")
    fig.subplots_adjust( hspace = 0 )
    plt.savefig(savename); plt.clf()
    
    return None

## Function: Peak finding function! ##       
def Find_Peaks( wav, spec, specsig, peaksnr = 5, minsep = 0.5, pwidth = 10 ):
    # peaksnr sets SNR needed for peak, minsep dictates how far apart peaks must be, pwidth is for fitting each found peak to get center
    
    # Find peaks using the cwt routine from scipy.signal, parameters are okay I think
    peaks = signal.find_peaks_cwt( spec, np.arange( 2, 4 ), min_snr = peaksnr, noise_perc = 25.0 )
    
    # Offset from start/end of spectrum by rough width of a peak
    peaks = peaks[ (peaks > pwidth) & (peaks < len(spec) - pwidth) ]
    
    pixcent = np.array([])
    wavcent = np.array([])
        
    for peak in peaks:
        # Try to perform a fit to the peak to get the center value of the line, and assign value from preliminary wavelength solution
        
        xi   = wav[peak - pwidth:peak + pwidth]
        yi   = spec[peak - pwidth:peak + pwidth]
        sigi = specsig[peak - pwidth:peak + pwidth]
        inds = np.arange( len(xi), dtype = float )
        
        p0       = [ yi[9], np.median( inds ), 0.9, np.median( spec ) ]
        lowerbds = [ p0[3], p0[1] - 2.0, 0.3, 0.0  ]
        upperbds = [ None, p0[1] + 2.0, 1.5, None ]
        parinfo  = [ { 'limits': ( lowerbds[i], upperbds[i] ) } for i in range( len(p0) ) ]

        try:
            # Try the fit (it might not work hah)
            params, res = mpyfit.fit( Least, p0, ( inds, yi, sigi, GaussModel ), parinfo = parinfo ) # Fit peak with a 1D Gaussian

            pixval  = peak - pwidth + params[1]    # Get central pixel and wavelength values!
            pixcent = np.append( pixcent, pixval )
            
            ceiling = np.ceil( pixval ).astype(int)
            floor   = np.floor( pixval ).astype(int)
            slope   = ( wav[ceiling] - wav[floor] ) / ( ceiling - floor )
            wavval  = wav[floor] + slope * ( pixval - floor )
            wavcent = np.append( wavcent, wavval )

        except RuntimeError: # If fit fails!
            pixval = 0 # Just a throwaway command to keep the loop going
    
    vals = spec[pixcent.astype(int)]
    oks  = np.ones( len(pixcent), dtype = bool )
    
    for i in range( len(wavcent) ):
        # Get rid of lines that are too close to each other, and only keep the highest peak of the ones too close together
        dist  = np.absolute( wavcent - wavcent[i] )
        close = np.where( dist <= minsep )[0]
        small = np.where( vals[close] < np.max( vals[close] ) )[0]
        if len(small) != 0: oks[close[small]] = 0

    return pixcent[oks], wavcent[oks]

## Function: Smoothing arc spectra ## 
def Smooth_Spec( spec, specsig ):
    
    smoothed  = spec.copy()    # Smoothed spectrum
    normfilt  = spec.copy()    # Normalized "continuum"
    smoothsig = specsig.copy() # Smoothed errors
    allfilt   = spec.copy()
    
    for i in range( spec.shape[0] ):
        for j in range( spec.shape[1] ):
            cutspec         = spec[i,j].copy()
            cut             = spec[i,j] >= np.percentile( spec[i,j], 90.0 )
            cutspec[cut]    = np.percentile( spec[i,j], 75.0 )
            filtered        = signal.savgol_filter( cutspec, 101, 3 )
            smoothed[i,j]  /= filtered
            smoothsig[i,j] /= filtered
            normfilt[i,j]   = filtered / np.max( filtered )
            allfilt[i,j]    = filtered

    return smoothed, smoothsig, normfilt

## Function: Getting the shift from the preliminary wavelength solution ## THINK ABOUT THIS!
def Get_Shift( cube, Conf ):
    # Function to determine if there's a shift in the orders extracted
    comp   = pickle.load( open( Conf.codedir + 'normfilt.pkl', 'rb' ) )

    shifts = []
    for j in np.arange( 5, cube.shape[0] - 4 ):
        difs = []
        for i in range( comp.shape[0] ):
            dif = np.absolute( comp[i] - cube[j] )
            difs.append( np.average( dif ) )
        mindif = np.argmin( difs )
        shifts.append( mindif - j )
    
    test, counts = np.unique( shifts, return_counts = True )
    shift        = test[np.argmax(counts)]

    return shift

## Function: Fitting the wavelength solution! ##
def Fit_WavSol( wav, spec, specsig, THARcat, path, THAR, Conf, snr = 5, minsep = 0.5, plots = True ):
    
    wavsol = np.zeros( len(spec) ) # Initialize the wavelength solution for this order
    
    # Find peaks in the arc spectrum!
    pixcent, wavcent = Find_Peaks( wav, spec, specsig, peaksnr = snr, minsep = minsep )
    
    # Initialize kept/rejected peaks, and match to nearest in the THAR catalog
    keeps = { 'pix': np.array([]), 'wav': np.array([]), 'line': np.array([]) }
    rejs  = { 'pix': np.array([]), 'wav': np.array([]), 'line': np.array([]) }
    
    for i in range( len( wavcent ) ): # Find matches to peaks in the THAR catalogue
        dists    = np.absolute( THARcat - wavcent[i] )
        mindist  = np.argmin( dists )
        
        if dists[mindist] <= 1.0:
            keeps['pix']  = np.append( keeps['pix'], pixcent[i] )
            keeps['wav']  = np.append( keeps['wav'], wavcent[i] )
            keeps['line'] = np.append( keeps['line'], THARcat[mindist] )

    # Now actually do the fit!
    dofit  = True
    ploti  = 1
    cutoff = 4.0 / 0.67449 # Corrects MAD to become sigma
    
    while dofit:
        
        res = np.polyfit( keeps['pix'], keeps['line'], Conf.WavPolyOrd, full = True ) # Polynomial fit to wavelength solution
        
        wavparams  = res[0]
        fitresult  = res[1:]
        ptsfromfit = np.polyval( wavparams, keeps['pix'] )
        
        # Calculate residuals in wavelength and velocity
        wavresids  = ptsfromfit - keeps['line']
        velresids  = wavresids / keeps['line'] * 3e5
        resids     = { 'wav': wavresids, 'vel': velresids }
        medabsdev  = np.median( np.abs( np.abs( resids['wav'] ) - np.median( np.abs( resids['wav'] ) ) ) )
        
        # Determine which lines are outliers in wavelength and velocity residuals
        velcut = np.sum( np.abs(resids['vel']) >= 5.0 )
        
        torej  = ( np.abs( resids['wav'] ) >= cutoff * medabsdev ) # | ( keeps['pix'] < 512 ) | ( keeps['pix'] > 1536 )
        
        tokeep = np.logical_not( torej )
        numrej = np.sum( torej )
        
        if velcut > 0 and numrej != len(torej): # If there are points that are outliers!
            if numrej > 0: # If there are points to reject based on wavelength residual

                plotname = path + '/resids_round_' + str(ploti) + '.pdf'
                Plot_WavSol_Resids( resids, keeps['line'], cutoff, plotname, tokeep = tokeep, toreject = torej )

                pltwav = np.polyval( wavparams, np.arange( len( spec ) ) )
                plt.clf()
                plt.plot( pltwav, np.log10(spec), 'k-', lw = 1 )
                plt.plot( THAR['wav'], THAR['logspec'], 'r-', lw = 1 )
                plt.xlim( pltwav[0], pltwav[-1] )
                for peak in keeps['line']:
                    plt.axvline( x = peak, color = 'b', ls = ':', lw = 1 )
                plt.savefig(path + '/rejplots/fullspec_' + str(ploti) + '.pdf'); plt.clf()
                
                rejs['pix']  = keeps['pix'][torej]
                rejs['wav']  = keeps['wav'][torej]
                rejs['line'] = keeps['line'][torej]
                
                keeps['pix']  = keeps['pix'][tokeep]
                keeps['wav']  = keeps['wav'][tokeep]
                keeps['line'] = keeps['line'][tokeep]
                
                ploti += 1
                
            elif numrej == 0 and velcut > 0: # If there aren't points to reject via wavelength residual, but still velocity outliers
                cutoff = cutoff - 1.0 / 0.67449
                
            else: # Honestly a little unsure what this is doing....
                if Conf.verbose: print( 'There is something seriously wrong.\n' )
                if Conf.verbose: print( 'There are points > 0.2 km/s, but none are found to be rejected. FIX' )
                flag = True
                if plots:
                    plotname = path + '/resids_round_' + str(ploti) + '_flag.pdf'
                    Plot_WavSol_Resids( resids, keeps['line'], cutoff, plotname )
                break

        elif numrej == len(torej): # If it wants to reject all the points! That's bad!
            if plots:
                plotname = path + '/resids_round_' + str(ploti) + '.pdf'
                Plot_WavSol_Resids( resids, keeps['line'], cutoff, plotname, toreject = torej )
            flag = True
            dofit = False            

        else: # If it all works out fine!
            if plots:
                plotname = path + '/resids_round_' + str(ploti) + '.pdf'
                Plot_WavSol_Resids( resids, keeps['line'], cutoff, plotname, tokeep = tokeep )
            flag = False
            dofit = False

    if fitresult[0].size == 0: flag = True # Basically if there aren't enough peaks and the fit isn't well constrained
    
    wavsol = np.polyval( wavparams, np.arange( len(spec) ) ) # Full wavelength solution for the order

    return wavsol, wavparams, keeps, rejs, flag

## Function: Post processing of wavelength solution, to correct flagged bad orders ##
def WavSol_PostProc( wavsol, wavparams, badorders ):
    
    for o in range( badorders.shape[1] ):
        
        if badorders[:,o].sum() > 0:
            
            if badorders[:,o].sum() == badorders.shape[0]:
                newpars = np.median( wavparams[:,o], axis = 0 )
            else:
                newpars = np.median( wavparams[~badorders[:,o],o], axis = 0 )

            wavparams[badorders[:,o],o] = newpars
            wavsol[badorders[:,o],o] = np.polyval( newpars, np.arange( 2048 ) )
            
    return wavsol, wavparams

## Function: calls above functions to return the full arc wavelength solution ##
def Get_WavSol( Cube, CubeSig, Conf, plots = True, Frames = 'All', Orders = 'All' ):

    if not Conf.doArcWav:
        # If wavelength solution has already been found, read it in!
        FinalWavSol = pickle.load( open( Conf.rdir + 'wavsol.pkl', 'rb' ) )

    else:
        if not os.path.exists( Conf.rdir + 'wavcal' ):
            # Make the directory containing all of the wavelength calibration information
            os.mkdir( Conf.rdir + 'wavcal' )

        # Set up loops for the frames and orders, in case don't wanna do them all
        if Frames == 'All': frameloop = range( Cube.shape[0] )
        else:               frameloop = Frames
        
        if Orders == 'All': orderloop = range( Cube.shape[1] )
        else:               orderloop = Orders

        # Read in the preliminary wavelength solution
        roughsol = pickle.load( open( Conf.codedir + Conf.PrelimWav, 'rb' ) )

        # Find the shift in the orders extracted (if there is one) think about this
        smoothcube, smoothsig, filtcube = Smooth_Spec( Cube, CubeSig )
        # orderdif                        = Get_Shift( filtcube[0], Conf )
        
        orderdif = 0

        # if orderdif < 0 or orderdif + Cube.shape[1] > roughsol.shape[0]:
        #     print( orderdif )
        #     pdb.set_trace()
        #     raw_input( 'Problem with number of orders found.\n' )

        # Initialize arrays for the full wavelength solution, and the wavelength fit parameters
        FullWavSol  = np.zeros( ( Cube.shape[0], Cube.shape[1], Cube.shape[2] ) )
        FullParams  = np.zeros( ( Cube.shape[0], Cube.shape[1], Conf.WavPolyOrd + 1 ) )
        
        # Read in ThAr information -- comp spectrum and line list
        THAR            = { }
        THARcalib       = fits.open( Conf.codedir + 'thar_photron.fits' )[0]
        THAR['spec']    = THARcalib.data
        THAR['wav']     = np.arange( len(THAR['spec']) ) * THARcalib.header['CDELT1'] + THARcalib.header['CRVAL1']
        THAR['logspec'] = np.log10( THAR['spec'] )
        THAR['lines']   = pd.read_table( Conf.codedir + 'ThAr_list.txt', delim_whitespace = True ).wav.values

        # Collect list of bad orders -- will use this for post processing some bad orders.
        badorders = np.zeros( ( Cube.shape[0], Cube.shape[1] ), dtype = bool )
        
        for frame in frameloop:
            # Loop through arc frames to solve wavlength for
            
            framepath = Conf.rdir + 'wavcal/arcframe_' + str( frame ) # Set up frame directories and what not
            if not os.path.exists( framepath ):
                os.mkdir( framepath )

            for order in orderloop:
                # Loop through orders to solve wavelength for
                
                orderpath = framepath + '/order_' + str(order) # Set up order directories and what not
                if os.path.exists( orderpath ):
                    for f in glob.glob( orderpath + '/*.*' ): os.remove(f)
                else:
                    os.mkdir( orderpath )

                if os.path.exists( orderpath + '/rejplots' ):
                    for f in glob.glob( orderpath + '/rejplots/*' ): os.remove(f)
                else:
                    os.mkdir( orderpath + '/rejplots' )

                arcspec    = smoothcube[frame,order,:]
                arcsigma   = smoothsig[frame,order,:]
                prelimsol  = roughsol[order+orderdif,:]

                logarcspec = np.log10( arcspec )

                wavsol, params, kept, rejs, flag = Fit_WavSol( prelimsol, arcspec, arcsigma, THAR['lines'], orderpath, THAR, Conf, plots = plots )
            
                if flag:
                    badorders[frame,order] = True
                
                pickle.dump( kept, open( orderpath + '/peakskept_ord_' + str(order) + '.pkl', 'wb' ) )
                pickle.dump( rejs, open( orderpath + '/peaksrej_ord_' + str(order) + '.pkl', 'wb' ) )
            
                # Do plots for this order -- full spectrum plot with peaks, and window plot for zoom in
                Plot_Order_Wavsol( wavsol, kept['line'], logarcspec, THAR, orderpath, frame, order )

                FullWavSol[frame,order] = wavsol
                FullParams[frame,order] = params
                
            pickle.dump( FullWavSol[frame], open( framepath + '/wavsol_' + str(frame) + '.pkl', 'wb' ) )
            pickle.dump( FullParams[frame], open( framepath + '/wavparams_' + str(frame) + '.pkl', 'wb' ) )
            
        for f in range( badorders.shape[0] ): print( 'Bad orders for frame', str(f), np.where( badorders[f] )[0] )
        
        # Now to fix those bad orders!
        
        FinalWavSol, FinalParams = WavSol_PostProc( FullWavSol, FullParams, badorders )
        
        pickle.dump( FinalWavSol, open( Conf.rdir + 'wavsol.pkl', 'wb' ) )
        pickle.dump( FinalParams, open( Conf.rdir + 'wavparams.pkl', 'wb' ) )
        pickle.dump( badorders, open( Conf.rdir + 'badorders_arcwav.pkl', 'wb' ) )
    
    return FinalWavSol

## Function: Interpolation from arc wavelength solution to object exposures ##
def Interpolate_Obj_WavSol( wavsol, info, arcinds, objinds, Conf ):

    if not Conf.doObjWav: # If done, just read in previous object wavelength
        Wsol_Obj = pickle.load( open( Conf.rdir + 'objwavsol.pkl', 'rb' ) )

    else:
        times    = ( info.UTdate.values + 'T' + info.UT.values ).astype( 'S22' ) # Get UT of exposures from headstrip info
        times    = [ t.replace( '/', '-' ) for t in times ]
        juldays  = Time( times, format = 'isot', scale = 'utc' ).jd
        arctime  = juldays[arcinds]
        objtime  = juldays[objinds]

        Wsol_Obj = np.zeros( ( len(objtime), wavsol.shape[1], wavsol.shape[2] ) )

        for i in range( len( objtime ) ):
            deltime = objtime[i] - arctime

            if np.max( arctime ) < objtime[i] or np.min( arctime ) > objtime[i]: # If it isn't bounded by arcs, just use the closest one!
                print( 'Target obs', str( i + 1 ), 'not bounded by arcs, using closest arc.' )
                closest     = np.argmin( np.abs( deltime ) )
                Wsol_Obj[i] = wavsol[closest]
            else: # Do the interpolation between two closest arcs (Maybe do like a full on interpolation?)
                bef = np.where( deltime > 0 )[0][-1]
                aft = np.where( deltime < 0 )[0][0]
                Wsol_Obj[i] = wavsol[bef] + ( wavsol[aft] - wavsol[bef] ) / ( arctime[aft] - arctime[bef] ) * ( objtime[i] - arctime[bef] )

        pickle.dump( Wsol_Obj, open( Conf.rdir + 'objwavsol.pkl', 'wb' ) )

    return Wsol_Obj

#############################################

### Section: Ancillary functions! ###

## Function: Continuum fit the extracted object spectra ##
def doContinuumFit( spec, sigspec, Conf, obj_filename ):
    
    thecont    = spec.copy()
    spec_cf    = spec.copy()
    sigspec_cf = spec.copy()

    for f in range( spec.shape[0] ): # Loop through the frames
        for o in range( spec.shape[1] ): # Loop through the orders
            
            if Conf.verbose:
                print( 'Continuum fitting frame', str(f+1), 'of', str(spec.shape[0]), 'and order', str(o+1), 'of', str(spec.shape[1]) )
            
            thecont[f,o] = getContinuum( spec[f,o], maxiter = 10, lower = 0.3, upper = 1.5, bkspace = 300, nord = 3 )

            spec_cf[f,o]    = spec[f,o] / thecont[f,o]
            sigspec_cf[f,o] = sigspec[f,o] / thecont[f,o]
            
    pickle.dump( thecont, open( Conf.rdir + 'cont' + obj_filename + '.pkl', 'wb' ) )
    pickle.dump( spec_cf, open( Conf.rdir + 'contfitspec' + obj_filename + '.pkl', 'wb' ) )
    pickle.dump( sigspec_cf, open( Conf.rdir + 'contfitsigspec' + obj_filename + '.pkl', 'wb' ) )
            
    return thecont, spec_cf, sigspec_cf
