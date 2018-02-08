import DMK_go_coude as Fns
import numpy as np
import pandas as pd
import os, pickle, pdb

class Configs():
    def __init__( self ):
        self.dir     = os.getenv("HOME") + '/Research/YMG/coude_data/20171102/'
        self.rdir    = self.dir + 'reduction/'
        self.codedir = os.getenv("HOME") + '/codes/coudereduction/'

        self.CalsDone   = True
        self.TraceDone  = False
        self.ArcExDone  = True
        self.ObjExDone  = True
        self.ArcWavDone = True
        self.ObjWavDone = True

        self.PlotsOn    = False

        self.DarkCurVal = 0.0
        self.MedCut     = 90.0

        self.InfoFile   = 'headstrip.csv'

Conf = Configs()

if not os.path.exists( Conf.rdir ):
    os.mkdir( Conf.rdir )
    os.mkdir( Conf.rdir + 'plots/' )

print '\nYou are reducing directory ' + Conf.dir + '\n'
raw_input( 'If that isn\'t right ctrl-c outta this! Otherwise just hit enter.\n' )

os.chdir(Conf.dir)

Fns.Header_Info( Conf )

FileInfo = pd.read_csv( Conf.InfoFile )

DarkCube = FileInfo.ExpTime * Conf.DarkCurVal

BiasInds = np.where( FileInfo.Type == 'zero' )[0]
FlatInds = np.where( FileInfo.Type == 'flat' )[0]
ArcInds  = np.where( (FileInfo.Type == 'comp') & ( (FileInfo.Object == 'Thar') | (FileInfo.Object == 'THAR') | (FileInfo.Object == 'A') ) )[0]
ObjInds  = np.where( (FileInfo.Type == 'object') & (FileInfo.Object != 'solar') & (FileInfo.Object != 'SolPort') & (FileInfo.Object != 'solar port') & (FileInfo.Object != 'Solar Port') )[0]

SuperBias, FlatField = Fns.Basic_Cals( FileInfo.File[BiasInds].values, FileInfo.File[FlatInds].values, Conf )

BPM = Fns.Make_BPM( SuperBias, FlatField, 99.9, Conf )

ArcCube, ArcSNR, ObjCube, ObjSNR = Fns.Return_Cubes( ArcInds, ObjInds, FileInfo, DarkCube, SuperBias, FlatField, BPM )

MedTrace, FitTrace = Fns.Get_Trace( FlatField, ObjCube, Conf )
pdb.set_trace()
# Just for now to make sure same orders are extracted.
FitTrace = FitTrace[:58]

if Conf.ArcExDone:
    wspec     = pickle.load( open( Conf.rdir + 'extracted_wspec.pkl', 'rb' ) )
    sig_wspec = pickle.load( open( Conf.rdir + 'extracted_sigwspec.pkl', 'rb' ) )
else:
    wspec, sig_wspec = Fns.Extractor( ArcCube, ArcSNR, FitTrace, quick = True, arc = True, nosub = True )
    pickle.dump( wspec, open( Conf.rdir + 'extracted_wspec.pkl', 'wb' ) )
    pickle.dump( sig_wspec, open( Conf.rdir + 'extracted_sigwspec.pkl', 'wb' ) )

if Conf.ObjExDone:
    spec      = pickle.load( open( Conf.rdir + 'extracted_spec.pkl', 'rb' ) )
    sig_spec  = pickle.load( open( Conf.rdir + 'extracted_sigspec.pkl', 'rb' ) )
else:
    spec, sig_spec   = Fns.Extractor( ObjCube, ObjSNR, FitTrace, quick = False, arc = False, nosub = False )
    pickle.dump( spec, open( Conf.rdir + 'extracted_spec.pkl', 'wb' ) )
    pickle.dump( sig_spec, open( Conf.rdir + 'extracted_sigspec.pkl', 'wb' ) )
#    pickle.dump( spec, open( Conf.rdir + 'extracted_spec_SNRtest.pkl', 'wb' ) )
#    pickle.dump( sig_spec, open( Conf.rdir + 'extracted_sigspec_SNRtest.pkl', 'wb' ) )

pdb.set_trace()

wspec      = wspec[:,::-1,:]
sig_wspec  = sig_wspec[:,::-1,:]
spec       = spec[:,::-1,:]
sig_spec   = sig_spec[:,::-1,:]

arcwavsol = Fns.Get_WavSol( wspec, sig_wspec, Conf )
objwavsol = Fns.Interpolate_Obj_WavSol( arcwavsol, FileInfo, ArcInds, ObjInds, Conf )

Fns.Spec_Plots( objwavsol, spec, sig_spec, spec / sig_spec, 20.0, 8, Conf.rdir )
