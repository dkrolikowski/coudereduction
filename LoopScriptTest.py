import DMK_go_coude as Fns
import numpy as np
import pandas as pd
import os, readcol, pickle, pdb

nightarr = [ '20171025', '20171026', '20171027', '20171028', '20171029', '20171030', '20171031', '20171101' ]

for night in nightarr:
    
    class Configs():
        def __init__( self ):
            self.dir     = os.getenv("HOME") + '/Research/YMG/coude_data/' + night + '/'
            self.rdir    = self.dir + 'reduction/'
            self.codedir = os.getenv("HOME") + '/codes/coudereduction/'
    
            self.CalsDone   = True
            self.TraceDone  = True
            self.ArcExDone  = True
            self.ObjExDone  = False
            self.ArcWavDone = True
            self.ObjWavDone = True
    
            self.PlotsOn    = False
    
            self.DarkCurVal = 0.0
            self.MedCut     = 95.0
    
            self.InfoFile   = 'headstrip.csv'
    
    Conf = Configs()
    
    if not os.path.exists( Conf.rdir ):
        os.mkdir( Conf.rdir )
        os.mkdir( Conf.rdir + 'plots/' )
    
    print '\nYou are reducing directory ' + Conf.dir + '\n'
    
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
    
