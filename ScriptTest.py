import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import DMK_go_coude as Fns
import numpy as np
import pandas as pd
import os, readcol, pickle, pdb

from astropy.io import fits 

dir = os.getenv("HOME") + '/Research/YMG/coude_data/20161205/'
rdir = dir + 'reduction/'
if os.getenv("HOME").split('/')[-1] == 'dmk2347':
    codedir = os.getenv("HOME") + '/codes/coudereduction/'
else:
    codedir = os.getenv("HOME") + '/Research/Codes/coudereduction/'

plotson     = False
CalsDone    = True
TraceDone   = True
ExtractDone = True

if not os.path.exists( rdir ):
    os.mkdir( rdir )
    os.mkdir( rdir + 'plots/' )

os.chdir(dir)

DarkCurVal = 0.0

InfoFile = 'headstrip.csv'
Fns.Header_Info( dir, InfoFile )

#FileInfo = readcol.readcol( InfoFile, fsep = ',', asRecArray = True )
FileInfo = pd.read_csv( InfoFile )

DarkCube = FileInfo.ExpTime * DarkCurVal

BiasInds = np.where( FileInfo.Type == 'zero' )[0]
FlatInds = np.where( FileInfo.Type == 'flat' )[0]
ArcInds  = np.where( (FileInfo.Type == 'comp') & ( (FileInfo.Object == 'Thar') | (FileInfo.Object == 'THAR') | (FileInfo.Object == 'A') ) )[0]
ObjInds  = np.where( (FileInfo.Type == 'object') & (FileInfo.Object != 'solar') & (FileInfo.Object != 'SolPort') & (FileInfo.Object != 'solar port') )[0]

SuperBias, FlatField = Fns.Basic_Cals( FileInfo.File[BiasInds].values, FileInfo.File[FlatInds].values, CalsDone, rdir, plots = plotson )

BPM = Fns.Make_BPM( SuperBias, FlatField, 99.9, rdir, plots = plotson )

RdNoise  = FileInfo.rdn[ArcInds] / FileInfo.gain[ArcInds]
DarkCur  = DarkCube[ArcInds] / FileInfo.gain[ArcInds]
ArcCube, ArcSNR = Fns.Make_Cube( FileInfo.File[ArcInds].values, RdNoise.values, DarkCur.values, Bias = SuperBias )

RdNoise  = FileInfo.rdn[ObjInds] / FileInfo.gain[ObjInds]
DarkCur  = DarkCube[ObjInds] / FileInfo.gain[ObjInds]
ObjCube, ObjSNR = Fns.Make_Cube( FileInfo.File[ObjInds].values, RdNoise.values, DarkCur.values, Bias = SuperBias, Flat = FlatField, BPM = BPM )

MedCut = 95.0
MedTrace, FitTrace = Fns.Get_Trace( FlatField, ObjCube, MedCut, rdir, TraceDone, plots = plotson )

if ExtractDone:
    wspec     = pickle.load( open( rdir + 'extracted_wspec.pkl', 'rb' ) )
    sig_wspec = pickle.load( open( rdir + 'extracted_sigwspec.pkl', 'rb' ) )
else:
    wspec, sig_wspec = Fns.extractor( ArcCube, ArcSNR, FitTrace, quick = True, arc = True, nosub = True )
    pickle.dump( wspec, open( rdir + 'extracted_wspec.pkl', 'wb' ) )
    pickle.dump( sig_wspec, open( rdir + 'extracted_sigwspec.pkl', 'wb' ) )

wspec      = wspec[:,::-1,:]
sig_wspec  = sig_wspec[:,::-1,:]
# spec       = spec[:,::-1,:]
# sig_spec   = sig_spec[:,::-1,:]

sols, params = Fns.Get_WavSol( wspec, rdir, codedir, Orders = np.arange( 0, 57 ) )
pickle.dump( sols, open( rdir + 'wavsol.pkl', 'wb' ) )
pickle.dump( params, open( rdir + 'wavparams.pkl', 'wb' ) )
