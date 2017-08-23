import DMK_go_coude as Fns
import numpy as np
import pandas as pd
import os, readcol, pickle, pdb

class Configs():
    def __init__( self ):
        self.dir  = os.getenv("HOME") + '/Google Drive/YMG/coude_data/20140321/'
        self.rdir = self.dir + 'reduction/'
        if os.getenv("HOME").split('/')[-1] == 'dmk2347':
            self.codedir = os.getenv("HOME") + '/codes/coudereduction/'
        else:
            self.codedir = os.getenv("HOME") + '/Research/Codes/coudereduction'

        self.CalsDone    = True
        self.TraceDone   = True
        self.ExtractDone = False

        self.PlotsOn     = False

        self.DarkCurVal  = 0.0
        self.MedCut      = 95.0

        self.InfoFile    = 'headstrip.csv'

Conf = Configs()

if not os.path.exists( Conf.rdir ):
    os.mkdir( Conf.rdir )
    os.mkdir( Conf.rdir + 'plots/' )

print '\nYou are reducing directory ' + Conf.dir + '\n'
raw_input( 'If that isn\'t right ctrl-c outta this! Otherwise just hit enter.\n' )

os.chdir(Conf.dir)

Fns.Header_Info( Conf.dir, Conf.InfoFile )

FileInfo = pd.read_csv( Conf.InfoFile )

DarkCube = FileInfo.ExpTime * Conf.DarkCurVal

BiasInds = np.where( FileInfo.Type == 'zero' )[0]
FlatInds = np.where( FileInfo.Type == 'flat' )[0]
ArcInds  = np.where( (FileInfo.Type == 'comp') & ( (FileInfo.Object == 'Thar') | (FileInfo.Object == 'THAR') | (FileInfo.Object == 'A') ) )[0]
ObjInds  = np.where( (FileInfo.Type == 'object') & (FileInfo.Object != 'solar') & (FileInfo.Object != 'SolPort') & (FileInfo.Object != 'solar port') )[0]

SuperBias, FlatField = Fns.Basic_Cals( FileInfo.File[BiasInds].values, FileInfo.File[FlatInds].values, Conf )

BPM = Fns.Make_BPM( SuperBias, FlatField, 99.9, Conf )

ArcCube, ArcSNR, ObjCube, ObjSNR = Fns.Return_Cubes( ArcInds, ObjInds, FileInfo, DarkCube, SuperBias, FlatField, BPM )

MedTrace, FitTrace = Fns.Get_Trace( FlatField, ObjCube, Conf )

if Conf.ExtractDone:
    wspec     = pickle.load( open( Conf.rdir + 'extracted_wspec.pkl', 'rb' ) )
    sig_wspec = pickle.load( open( Conf.rdir + 'extracted_sigwspec.pkl', 'rb' ) )
    spec      = pickle.load( open( Conf.rdir + 'extracted_spec.pkl', 'rb' ) )
    sig_spec  = pickle.load( open( Conf.rdir + 'extracted_sigspec.pkl', 'rb' ) )
else:
    wspec, sig_wspec = Fns.extractor( ArcCube, ArcSNR, FitTrace, quick = True, arc = True, nosub = True )
    spec, sig_spec   = Fns.extractor( ObjCube, ObjSNR, FitTrace, quick = False, arc = False, nosub = False )
    pickle.dump( wspec, open( Conf.rdir + 'extracted_wspec.pkl', 'wb' ) )
    pickle.dump( sig_wspec, open( Conf.rdir + 'extracted_sigwspec.pkl', 'wb' ) )
    pickle.dump( spec, open( Conf.rdir + 'extracted_spec.pkl', 'wb' ) )
    pickle.dump( sig_spec, open( Conf.rdir + 'extracted_sigspec.pkl', 'wb' ) )

wspec      = wspec[:,::-1,:]
sig_wspec  = sig_wspec[:,::-1,:]
# spec       = spec[:,::-1,:]
# sig_spec   = sig_spec[:,::-1,:]

sols, params = Fns.Get_WavSol( wspec, Conf, Orders = [0] )
#pickle.dump( sols, open( rdir + 'wavsol.pkl', 'wb' ) )
#pickle.dump( params, open( rdir + 'wavparams.pkl', 'wb' ) )
