import DMK_go_coude as Fns
import os, readcol
import pandas as pd

## Set up directories for code and data

dir = '/Users/dmk2347/Research/YMG/coude_data/20140321/'
rdir = dir + 'reduction/oldred/'
codedir = '/Users/dmk2347/codes/tullcoude/'

if not os.path.exists( rdir ):
    os.mkdir( rdir )
    os.mkdir( rdir + '/plots' )

## Parameters for reduction

# Run header_info to create headstrip file
InfoFile    = 'headstrip.csv'
Fns.Header_Info( dir, InfoFile )

# True - Make master bias and flat, False - read them in
CalsDone    = True
# True - Show bad pixel mask
ShowBPM     = False
# True - Read in all arc/object data, False - if extractions are done
ReadCubes   = True
# True - Trace is done
TraceDone   = False
# True - data frames are extracted
ExtractDone = True
# True - Quick extraction
ExtractType = True
# True - Arcs extracted
ArcsExtract = True
# True - Wavelength calibration done
WCalDone    = False
# True - Interpolate wavelength solutions onto object timestamp
WInterpDone = False
# True - Continuum subtract/correct the spectra
FlattenDone = True

plotson = True

##### !!!!! Unsure about this
# Set the dark current per s of exposure
DarkCurVal = 0.0

print 'We are reducing directory ' + dir
raw_input( 'Hit enter if correct to continue with reduction:' )

os.chdir( dir ) # Switch to the data directory

## Read in information for data files from headstrip file

FileInfo  = readcol.readcol( InfoFile, fsep = ',', asRecArray = True )
DarkCube  = FileInfo.ExpTime * DarkCurVal

# Determine which files are bias/flat/arc/obj
BiasInds = np.where( FileInfo.Type == 'zero' )[0]
FlatInds = np.where( FileInfo.Type == 'flat' )[0]
ArcInds  = np.where( (FileInfo.Type == 'comp') & ( (FileInfo.Object == 'Thar') | (FileInfo.Object == 'THAR') | (FileInfo.Object == 'A') ) )[0]
ObjInds  = np.where( (FileInfo.Type == 'object') & (FileInfo.Object != 'solar') & (FileInfo.Object != 'SolPort') )[0]

## Do basic calibrations
SuperBias, FlatField = Fns.Basic_Cals( FileInfo.File[BiasInds], FileInfo.File[FlatInds], CalsDone, rdir, plots = plotson )

## Make a first mask of bad pixels from the Flat and Bias
BPM = Fns.Make_BPM( SuperBias, FlatField, 99.9, ShowBPM )

## Read in objects/arcs and bias/flat correct them

if ReadCubes == True:
    print 'Reading Arc Files'
    RdNoise  = FileInfo.rdn[ArcInds] / FileInfo.gain[ArcInds]
    DarkCur  = DarkCube[ArcInds] / FileInfo.gain[ArcInds]
    ArcCube, ArcSNR = Fns.Make_Cube( FileInfo.File[ArcInds], RdNoise, DarkCur, Bias = SuperBias )

    print 'Reading Object Files'
    RdNoise  = FileInfo.rdn[ObjInds] / FileInfo.gain[ObjInds]
    DarkCur  = DarkCube[ObjInds] / FileInfo.gain[ObjInds]
    ObjCube, ObjSNR = Fns.Make_Cube( FileInfo.File[ObjInds], RdNoise, DarkCur, Bias = SuperBias, Flat = FlatField, BPM = BPM )
    
## Perform preliminary trace on flat field, and then individual trace on every object

OrderStart = -32

MedTrace, FitTrace = Fns.Get_Trace( FlatField, ObjCube, OrderStart, rdir, TraceDone )

## Extract the spectra

if ExtractDone == False:
    Spec_Cube, Spec_Sig = Fns.Extractor( ObjCube, ObjSNR, MedTrace, quick = ExtractType, arc = False, nosub = False, ExtractDone )
    pickle.dump( Spec_Cube, open( rdir + 'extracted_spec.pkl', 'wb' ) )
    pikcle.dump( Spec_Sig, open( rdir + 'extracted_specsig.pkl', 'wb' ) )


