##### Script to run the coude reduction, not the most elegant interface for running the code, but it gets the job done

from __future__ import print_function

import tullcoude_reduce_fns as Fns
import numpy as np
import pandas as pd
import os, pickle, pdb

# I often just manually set a list of nights, because I'm never running that many, but could read in a file
nightarr = [ 20210130, 20210131 ]

# Make sure the input night value is a string for purposes of reading in files
if not isinstance( nightarr[0], basestring ):
    nightarr = [ str(night) for night in nightarr ]

#################################################################################################################

# Looping through each night in the list to run the reduction
for night in nightarr:

    ########## Configuration of the reduction run ##########

    # Setting up the configurations/options for the reduction (could be in a separate file, that would be ideal)
    class Configs():

        def __init__( self ):

            # Set the relevant directory paths
            self.dir     = os.getenv("HOME") + '/Research/YMG/coude_data/' + night + '/' # Path to the data
            self.rdir    = self.dir + 'reduction/' # Name of the directory holding the reduction output
            self.codedir = os.getenv("HOME") + '/codes/coudereduction/' # Directory with coude reduction code and relevant ancillary files

            # Set which steps to do
            self.doCals   = False    # Extract and reduce calibration files
            self.doTrace  = False    # Do the trace!
            self.doArcEx  = False    # Extract thar spectra -- simple extraction
            self.doObjEx  = True    # Extract object spectra -- simple or full extraction
            self.doArcWav = False    # Determine thar spectra wavelength solutions
            self.doObjWav = False    # Apply thar wavelength solutions to object spectra
            self.doContFit  = True   # Continuum fit the object spectra

            # Other parameters for reduction steps
            self.CosmicSub  = False   # Create object spectral cube with cosmic ray subtraction
            self.ObjExType  = 'arc'  # Set the extraction method for objects: 'full' or 'arc', arc is just a simple pixel column sum
            self.verbose    = True    # Basically just have as much printing of what's going on to the terminal
            self.WavPolyOrd = 2       # Polynomial order for the wavelength solution fit
            self.cos_iters  = 2       # Set the number of iterations for the cosmic subtraction
            self.DarkCurVal = 0.0    # Value of the dark current
            self.BPMlimit   = 99.9   # Percentile to mark above as a bad pixel
            self.MedCut     = 85.0   # Flux percentile to cut at when making final trace using object spectra

            # File names for things that are needed
            self.InfoFile   = 'headstrip.csv'   # Name for the header info file
            self.PrelimWav  = 'prelim_wsol_new.pkl' # Name for the preliminary wavelength solution (initial guess)

    # Set the configurations to a variable to pass to functions
    Conf = Configs()

    # Make sure that the object extraction flag is allowed
    assert Conf.ObjExType in [ 'full', 'arc' ], 'Object extraction type must be either full or arc'

    ########## Directory and file set up ##########

    # Make sure that the reduction directory is there, and a subfolder for plots
    if not os.path.exists( Conf.rdir ):
        os.mkdir( Conf.rdir )
    if not os.path.exists( Conf.rdir + 'plots/' ):
        os.mkdir( Conf.rdir + 'plots/' )

    print( '\nYou are reducing directory', Conf.dir, 'Better be right!\n' )

    # Get into the night's directory!
    os.chdir( Conf.dir )

    # Create the header info file
    if not os.path.exists( Conf.dir + Conf.InfoFile ):
        Fns.Header_Info( Conf )
    FileInfo = pd.read_csv( Conf.InfoFile )

    # Get the file indices for the different file types: bias, flat, thar, and objects
    BiasInds = np.where( FileInfo.Type == 'zero' )[0] # Bias frames
    FlatInds = np.where( FileInfo.Type == 'flat' )[0] # Flat frames

    # Thar and objects are a bit more annoying, because they have been called many things in the headers, here are names to include/exclude

    # Header target names that have been used for ThAr/comp type
    arc_hdrnames    = [ 'Thar', 'ThAr', 'THAR', 'A' ]
    # Header target names that are objects but SHOULD not be included as objects
    notobj_hdrnames = [ 'solar', 'SolPort', 'solar port', 'Solar Port', 'test', 'SolarPort', 'Solport', 'solport', 'Sol Port', 'Solar Port Halpha' ]

    # Set a minimum exposure time for ThAr frames to include (has to be above 30 s, otherwise the blue orders aren't good)
    arc_exptime     = 30.0

    # Get the thar and object indices
    ArcInds    = np.where( np.logical_and( ( ( FileInfo.Type.values == 'comp' ) & ( FileInfo.ExpTime.values > arc_exptime ) ), np.any( [ FileInfo.Object == hdrname for hdrname in arc_hdrnames ], axis = 0 ) ) )[0]
#    ArcInds = np.where( np.logical_and( FileInfo.Type.values == 'comp', np.any( [ FileInfo.Object == hdrname for hdrname in arc_hdrnames ], axis = 0 ) ) )[0]
    ObjInds = np.where( np.logical_and( FileInfo.Type.values == 'object', np.all( [ FileInfo.Object != hdrname for hdrname in notobj_hdrnames ], axis = 0 ) ) )[0]

    ########## Get the image cubes set up: calibrations and objects ##########

    # Get dark current (I set this to 0)
    DarkCube = FileInfo.ExpTime * Conf.DarkCurVal

    # Get the combined bias and flat images, and a bad pixel mask from the combined bias
    SuperBias, FlatField, BPM = Fns.Basic_Cals( BiasInds, FlatInds, FileInfo, Conf )

    # Get the cubes containing ThAr and object images: as fluxes and S/N
    ArcCube, ArcSNR, ObjCube, ObjSNR = Fns.Return_Cubes( ArcInds, ObjInds, FileInfo, DarkCube, SuperBias, FlatField, BPM, Conf )

    ########## Get the trace! ##########

    # Get the trace (fitted trace is what I use)
    MedTrace, FitTrace = Fns.Get_Trace( FlatField['vals'], ObjCube, Conf )

    # Funky thing to make sure the same orders (at least mostly) are extracted every time. Here is an example of a hard-coded part of the reduction
    # Sometimes I find more orders, further in the red, but very low signal so often don't contain much of use
    FitTrace = FitTrace[:58]

    ########## Spectral extraction! ##########

    ### ThAr spectrum extraction ###

    # If marked as "already done" in the configuration setup, just read in the files
    if not Conf.doArcEx:
        wspec     = pickle.load( open( Conf.rdir + 'extracted_wspec.pkl', 'rb' ) )
        sig_wspec = pickle.load( open( Conf.rdir + 'extracted_sigwspec.pkl', 'rb' ) )

    # If marked as such in the configuration setup, perform the extraction!
    else:
        # Get the extracted spectrum and errors (using a simple extraction)
        wspec, sig_wspec = Fns.Extractor( ArcCube, ArcSNR, FitTrace, Conf, quick = True, arc = True, nosub = True )

        # Reverse the orders of the extracted spectrum so it goes from blue to red
        wspec     = wspec[:,::-1]
        sig_wspec = sig_wspec[:,::-1]

        # Output the extracted ThAr spectra and errors
        pickle.dump( wspec, open( Conf.rdir + 'extracted_wspec.pkl', 'wb' ) )
        pickle.dump( sig_wspec, open( Conf.rdir + 'extracted_sigwspec.pkl', 'wb' ) )

    ### Object extraction! ###

    # Set the suffix for objected exctrations (type of extraction and cosmic subtraction)
    obj_filename = ''
    if Conf.ObjExType == 'arc':
        obj_filename += '_quick' # Default without suffix is full extraction, if _quick is added then simple extraction
    if Conf.CosmicSub:
        obj_filename += '_cossub' # No suffix means no cosmic subtraction, _cossub (which is default) has cosmics subtracted

    # If marked as "already done" in the configuration setup, just read in the files
    if not Conf.doObjEx:
        spec      = pickle.load( open( Conf.rdir + 'extracted_spec' + obj_filename + '.pkl', 'rb' ) )
        sig_spec  = pickle.load( open( Conf.rdir + 'extracted_sigspec' + obj_filename + '.pkl', 'rb' ) )

    # If marked as such in the configuration setup, perform the extraction!
    else:
        if Conf.ObjExType == 'full': # Do full extraction if marked as such
            spec, sig_spec   = Fns.Extractor( ObjCube, ObjSNR, FitTrace, Conf, quick = False, arc = False, nosub = False )
        elif Conf.ObjExType == 'arc': # Do the quick/simple extraction if marked as such
            spec, sig_spec   = Fns.Extractor( ObjCube, ObjSNR, FitTrace, Conf, quick = True, arc = False, nosub = True )

        # Reverse the orders of the extracted spectrum so it goes from blue to red
        spec     = spec[:,::-1]
        sig_spec = sig_spec[:,::-1]

        # Output the extracted ThAr spectra and errors
        pickle.dump( spec, open( Conf.rdir + 'extracted_spec' + obj_filename + '.pkl', 'wb'  ) )
        pickle.dump( sig_spec, open( Conf.rdir + 'extracted_sigspec' + obj_filename + '.pkl', 'wb' ) )

    ########## Wavelength solution and calibration ##########

    # Get the wavelength solution from the ThAr spectra
    arcwavsol = Fns.Get_WavSol( wspec, sig_wspec, Conf )

    # Apply the wavelength solution to the object frames (interpolation)
    objwavsol = Fns.Interpolate_Obj_WavSol( arcwavsol, FileInfo, ArcInds, ObjInds, Conf )

    ########## Any additional steps thrown in ##########

    ### Continuum normalization ###

    if Conf.doContFit:
        print( 'Fitting the continuum!' )

        cont, spec_cf, sigspec_cf = Fns.doContinuumFit( spec, sig_spec, Conf, obj_filename )
