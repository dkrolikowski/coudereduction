import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import DMK_go_coude as Fns
import numpy as np
import os, readcol, pickle
import scipy.optimize as optim
import scipy.interpolate as interp

from astropy.io import fits 
from mpfit import mpfit
from scipy import signal

dir = os.getenv("HOME") + '/Research/YMG/coude_data/20161114/'
rdir = dir + 'reduction/'
#codedir = os.getenv("HOME") + '/codes/coudereduction/'
codedir = os.getenv("HOME") + '/Research/Codes/coudereduction/'

plotson = False

if not os.path.exists( rdir ):
    os.mkdir( rdir )

os.chdir(dir)

DarkCurVal = 0.0

InfoFile = 'headstrip.csv'
Fns.header_info( dir, InfoFile )

FileInfo = readcol.readcol( InfoFile, fsep = ',', asRecArray = True )

DarkCube = FileInfo.ExpTime * DarkCurVal

BiasInds = np.where( FileInfo.Type == 'zero' )[0]
FlatInds = np.where( FileInfo.Type == 'flat' )[0]
ArcInds  = np.where( (FileInfo.Type == 'comp') & ( (FileInfo.Object == 'Thar') | (FileInfo.Object == 'THAR') | (FileInfo.Object == 'A') ) )[0]
ObjInds  = np.where( (FileInfo.Type == 'object') & (FileInfo.Object != 'solar') & (FileInfo.Object != 'SolPort') )[0]

CalsDone = True
SuperBias, FlatField = Fns.Basic_Cals( FileInfo.File[BiasInds], FileInfo.File[FlatInds], CalsDone, rdir, plots = plotson )

ShowBPM = False
BPM = Fns.Make_BPM( SuperBias, FlatField, 99.9, ShowBPM )

RdNoise  = FileInfo.rdn[ArcInds] / FileInfo.gain[ArcInds]
DarkCur  = DarkCube[ArcInds] / FileInfo.gain[ArcInds]
ArcCube, ArcSNR = Fns.Make_Cube( FileInfo.File[ArcInds], RdNoise, DarkCur, Bias = SuperBias )

RdNoise  = FileInfo.rdn[ObjInds] / FileInfo.gain[ObjInds]
DarkCur  = DarkCube[ObjInds] / FileInfo.gain[ObjInds]
ObjCube, ObjSNR = Fns.Make_Cube( FileInfo.File[ObjInds], RdNoise, DarkCur, Bias = SuperBias, Flat = FlatField, BPM = BPM )

OrderStart = -32
TraceDone = True
MedCut = 90.0
MedTrace, FitTrace = Fns.Get_Trace( FlatField, ObjCube, OrderStart, MedCut, rdir, TraceDone, plots = plotson )

# wspec,sig_wspec = Fns.extractor( ArcCube,ArcSNR,FitTrace,quick=True,arc=True,nosub=True )
# pickle.dump( wspec, open( rdir + 'extracted_wspec.pkl', 'wb' ) )
# pickle.dump( sig_wspec, open( rdir + 'extracted_sigwspec.pkl', 'wb' ) )

wspec     = pickle.load( open( rdir + 'extracted_wspec.pkl', 'rb' ) )
sig_wspec = pickle.load( open( rdir + 'extracted_sigwspec.pkl', 'rb' ) )

# spec       = pickle.load(open(rdir+'extracted_spec_oldway.pkl','rb'))
# sig_spec   = pickle.load(open(rdir+'extracted_sigspec_oldway.pkl','rb'))
# wspec      = pickle.load(open(rdir+'extracted_wspec_oldway.pkl','rb'))
# sig_wspec  = pickle.load(open(rdir+'extracted_sigwspec_oldway.pkl','rb'))

wspec      = wspec[:,::-1,:]
sig_wspec  = sig_wspec[:,::-1,:]
# spec       = spec[:,::-1,:]
# sig_spec   = sig_spec[:,::-1,:]

roughsol = pickle.load( open( codedir + 'prelim_wsol.pkl', 'rb' ) )
sols, params = Fns.Get_WavSol( wspec, roughsol, rdir, codedir, Orders = [50] )
