from __future__ import print_function

import tullcoude_reduce_fns as Fns
import numpy as np
import pandas as pd
import os, pickle, pdb

##### Set the names of the directories you want to reduce! #####

#nightarr = [ 20181221, 20181222, 20190111, 20190112, 20190216, 20190217 ]
#nightarr = [ 20161205, 20161206, 20161219, 20161220, 20161222 ]
nightarr = [ 20200801, 20200802, 20200803, 20200804 ]

if not isinstance( nightarr[0], basestring ):
    nightarr = [ str(night) for night in nightarr ]
    
#################################################################################################################

##### Now we'll loop through all the nights and do the reduction! #####
    
for night in nightarr:
    
    ##### Set the configurations for the reduction! #####
    
    class Configs():
        
        def __init__( self ):
            
            ## Set directories ##
            self.dir     = os.getenv("HOME") + '/Research/YMG/coude_data/' + night + '/'
            self.rdir    = self.dir + 'reduction/'
            self.codedir = os.getenv("HOME") + '/codes/coudereduction/'
                
            ## Set which things to be done! ##
            self.doCals   = False    # Extract and reduce calibration files
            self.doTrace  = False    # Do the trace!
            self.doArcEx  = False    # Extract arc spectra -- simple extraction
            self.doObjEx  = True    # Extract object spectra -- full extraction
            self.doArcWav = False    # Determine arc spectra wavelength solutions
            self.doObjWav = False    # Apply wavelength solutions to object spectra
            
            ## Set other important parameters ##
            self.CosmicSub  = False   # Create object spectral cube with cosmic ray subtraction
            self.ObjExType  = 'full'  # Set the extraction method for objects: 'full' or 'arc'
            self.verbose    = True    # Basically just have as much printing of what's going on to the terminal
            self.WavPolyOrd = 2       # Polynomial order for the wavelength solution fit
            self.cos_iters  = 2       # Set the number of iterations for the cosmic subtraction
            
            self.InfoFile   = 'headstrip.csv'   # Name for the header info file
            self.PrelimWav  = 'prelim_wsol_new.pkl' # Name for the preliminary wavelength solution (initial guess)
            
            self.DarkCurVal = 0.0    # Value of the dark current
            self.BPMlimit   = 99.9   # Percentile to mark above as a bad pixel
            self.MedCut     = 85.0   # Flux percentile to cut at when making final trace using object spectra
            
            ## Other thing to do ##
            self.doContFit  = True   # Continuum fit the object spectra
    
    Conf = Configs() # Set the configurations to a variable to pass to functions
    
    assert Conf.ObjExType in [ 'full', 'arc' ], 'Object extraction type must be either full or arc'
    
    ##### Some directory and file setups! #####
        
    ## Make sure that the reduction directory actually exist! That would be a problem
    if not os.path.exists( Conf.rdir ):
        os.mkdir( Conf.rdir )
    if not os.path.exists( Conf.rdir + 'plots/' ):
        os.mkdir( Conf.rdir + 'plots/' )
    
    print( '\nYou are reducing directory', Conf.dir, 'Better be right!\n' )
    
    os.chdir( Conf.dir ) # Get into the night's directory!
    
    ## Create the header info file
    if not os.path.exists( Conf.dir + Conf.InfoFile ):
        Fns.Header_Info( Conf )
            
    FileInfo = pd.read_csv( Conf.InfoFile )
    
    ##### Set up file indices from header file #####
    
    BiasInds = np.where( FileInfo.Type == 'zero' )[0] ## Bias indicies
    FlatInds = np.where( FileInfo.Type == 'flat' )[0] ## Flat indicies
    
    ## Arcs and Objs are a bit annoying... ensure arc is called an arc and that an object isn't a solar port or test!
    arc_hdrnames    = [ 'Thar', 'ThAr', 'THAR', 'A' ]
    notobj_hdrnames = [ 'solar', 'SolPort', 'solar port', 'Solar Port', 'test', 'SolarPort', 'Solport', 'solport', 'Sol Port', 'Solar Port Halpha' ]
    
    arc_exptime     = 30.0
    
    ArcInds    = np.where( np.logical_and( ( ( FileInfo.Type.values == 'comp' ) & ( FileInfo.ExpTime.values > arc_exptime ) ), np.any( [ FileInfo.Object == hdrname for hdrname in arc_hdrnames ], axis = 0 ) ) )[0]
#    ArcInds = np.where( np.logical_and( FileInfo.Type.values == 'comp', np.any( [ FileInfo.Object == hdrname for hdrname in arc_hdrnames ], axis = 0 ) ) )[0]
    ObjInds = np.where( np.logical_and( FileInfo.Type.values == 'object', np.all( [ FileInfo.Object != hdrname for hdrname in notobj_hdrnames ], axis = 0 ) ) )[0]
    
    ##### Okay so now time for the calibrations and image cubes! Bias, flats, arcs, objects #####
        
    DarkCube = FileInfo.ExpTime * Conf.DarkCurVal ## Make a dark current array

    SuperBias, FlatField, BPM = Fns.Basic_Cals( BiasInds, FlatInds, FileInfo, Conf ) ## Make the master bias and flat field and bad pixel mask

    ## Make the image cubes! Outputs images and SNR images
    ArcCube, ArcSNR, ObjCube, ObjSNR = Fns.Return_Cubes( ArcInds, ObjInds, FileInfo, DarkCube, SuperBias, FlatField, BPM, Conf )
        
    ##### Now do the trace! This is basically all in the functions file #####
    
    MedTrace, FitTrace = Fns.Get_Trace( FlatField['vals'], ObjCube, Conf )
    
    ## Funky thing to make sure the same orders (at least mostly) are extracted every time. Might wanna change this later but...
    FitTrace = FitTrace[:58]
        
    ##### Extraction time! For both arcs and objects #####
    
    ## Arc extraction! 
    if not Conf.doArcEx: # If the extraction is already done, read in the files
        wspec     = pickle.load( open( Conf.rdir + 'extracted_wspec.pkl', 'rb' ) )
        sig_wspec = pickle.load( open( Conf.rdir + 'extracted_sigwspec.pkl', 'rb' ) )

    else: # If we need to do the extraction, do the extraction!
        wspec, sig_wspec = Fns.Extractor( ArcCube, ArcSNR, FitTrace, Conf, quick = True, arc = True, nosub = True )
        
        wspec     = wspec[:,::-1]     # Reverse orders so it goes from blue to red!
        sig_wspec = sig_wspec[:,::-1]
        
        pickle.dump( wspec, open( Conf.rdir + 'extracted_wspec.pkl', 'wb' ) )
        pickle.dump( sig_wspec, open( Conf.rdir + 'extracted_sigwspec.pkl', 'wb' ) )
    
    ## Object extraction!
    
    obj_filename = ''
    if Conf.ObjExType == 'arc': obj_filename += '_quick'
    if Conf.CosmicSub: obj_filename += '_cossub'
        
    if not Conf.doObjEx: # If the extraction is already done, read in the files
        
        spec      = pickle.load( open( Conf.rdir + 'extracted_spec' + obj_filename + '.pkl', 'rb' ) )
        sig_spec  = pickle.load( open( Conf.rdir + 'extracted_sigspec' + obj_filename + '.pkl', 'rb' ) )

    else: # If we need to do the extraction, do the extraction!

        if Conf.ObjExType == 'full':
            spec, sig_spec   = Fns.Extractor( ObjCube, ObjSNR, FitTrace, Conf, quick = False, arc = False, nosub = False )
        elif Conf.ObjExType == 'arc':
            spec, sig_spec   = Fns.Extractor( ObjCube, ObjSNR, FitTrace, Conf, quick = True, arc = False, nosub = True )

        spec     = spec[:,::-1]     # Reverse orders so it goes from blue to red!
        sig_spec = sig_spec[:,::-1]
            
        pickle.dump( spec, open( Conf.rdir + 'extracted_spec' + obj_filename + '.pkl', 'wb'  ) )
        pickle.dump( sig_spec, open( Conf.rdir + 'extracted_sigspec' + obj_filename + '.pkl', 'wb' ) )

    ##### Wavelength calibration now! #####
            
    arcwavsol = Fns.Get_WavSol( wspec, sig_wspec, Conf ) ## Arc wavelength solution
        
    objwavsol = Fns.Interpolate_Obj_WavSol( arcwavsol, FileInfo, ArcInds, ObjInds, Conf ) ## Object wavelength solution
    
    ##### Additional things to do! #####
    
    ## Continuum normalization
    
    if Conf.doContFit:
        print( 'Fitting the continuum!' )
        
        cont, spec_cf, sigspec_cf = Fns.doContinuumFit( spec, sig_spec, Conf, obj_filename )
