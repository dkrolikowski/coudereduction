import DMK_go_coude as Fns
import numpy as np
import pandas as pd
import os, readcol, pickle, pdb

#nightarr = [ '20171025', '20171026', '20171027', '20171028', '20171029', '20171030', '20171031', '20171101', '20171102' ]
#nightarr = [ '20171029', '20171030', '20171031', '20171101', '20171102' ]

#nightarr = [ 20161013, 20161014,20161015,20161016,20161114,20161115,20161116,20161117,20161118,20161119,20161120,20161121,20161122,20161123,20161124,20161125,20161126,20161127,20161205,20161206,20161207,20161208,20161219,20161220,20161221,20161222]
#nightarr = [ 20161013, 20161014,20161015,20161016,20161114,20161115,20161116,20161117,20161119,20161120,20161121,20161122,20161123,20161124,20161127,20161205,20161219,20161220 ]
#nightarr = [ 20161118, 20161125, 20161126,20161206,20161207,20161208,20161221,20161222 ] SKIPPED

## For wavcal
#nightarr = [ 20161014,20161015,20161016,20161114,20161115,20161116,20161117,20161119,20161120]
#nightarr = [20161121]
#nightarr = [20161122,20161123,20161124,20161127,20161205,20161219,20161220 ]

nightarr = [ 20180531 ]
nightarr = [ str(night) for night in nightarr ]

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
            self.ContDone   = True
    
            self.PlotsOn    = False
    
            self.DarkCurVal = 0.0
            self.MedCut     = 90.0
    
            self.InfoFile   = 'headstrip.csv'
    
    Conf = Configs()
    
    if not os.path.exists( Conf.rdir ):
        os.mkdir( Conf.rdir )
    if not os.path.exists( Conf.rdir + 'plots/' ):
        os.mkdir( Conf.rdir + 'plots/' )
    
    print '\nYou are reducing directory ' + Conf.dir + '\n'
    
    os.chdir( Conf.dir )
    
    Fns.Header_Info( Conf )
    
    FileInfo = pd.read_csv( Conf.InfoFile )
    
    DarkCube = FileInfo.ExpTime * Conf.DarkCurVal
    
    BiasInds = np.where( FileInfo.Type == 'zero' )[0]
    FlatInds = np.where( FileInfo.Type == 'flat' )[0]
    ArcInds  = np.where( (FileInfo.Type == 'comp') & ( (FileInfo.Object == 'Thar') | (FileInfo.Object == 'THAR') | (FileInfo.Object == 'A') ) )[0]
    ObjInds  = np.where( (FileInfo.Type == 'object') & (FileInfo.Object != 'solar') & (FileInfo.Object != 'SolPort') & (FileInfo.Object != 'solar port') & (FileInfo.Object != 'Solar Port') & (FileInfo.Object != 'test') )[0]
    
    SuperBias, FlatField = Fns.Basic_Cals( BiasInds, FlatInds, FileInfo, Conf )
    
    BPM = Fns.Make_BPM( SuperBias['vals'], FlatField['vals'], 99.9, Conf )
    
    ArcCube, ArcSNR, ObjCube, ObjSNR = Fns.Return_Cubes( ArcInds, ObjInds, FileInfo, DarkCube, SuperBias, FlatField, BPM )
    
    MedTrace, FitTrace = Fns.Get_Trace( FlatField['vals'], ObjCube, Conf )

    # Just for now to make sure same orders are extracted.
    FitTrace = FitTrace[:58]
    
    if Conf.ArcExDone:
        wspec     = pickle.load( open( Conf.rdir + 'extracted_wspec.pkl', 'rb' ) )
        sig_wspec = pickle.load( open( Conf.rdir + 'extracted_sigwspec.pkl', 'rb' ) )
    else:
        wspec, sig_wspec = Fns.Extractor( ArcCube, ArcSNR, FitTrace, quick = True, arc = True, nosub = True )
        
        wspec     = wspec[:,::-1]
        sig_wspec = sig_wspec[:,::-1]
        
        pickle.dump( wspec, open( Conf.rdir + 'extracted_wspec.pkl', 'wb' ) )
        pickle.dump( sig_wspec, open( Conf.rdir + 'extracted_sigwspec.pkl', 'wb' ) )
    
    if Conf.ObjExDone:
        spec      = pickle.load( open( Conf.rdir + 'extracted_spec.pkl', 'rb' ) )
        sig_spec  = pickle.load( open( Conf.rdir + 'extracted_sigspec.pkl', 'rb' ) )
    else:
        spec, sig_spec   = Fns.Extractor( ObjCube, ObjSNR, FitTrace, quick = False, arc = False, nosub = False )
#        spec, sig_spec   = Fns.Extractor( ObjCube, ObjSNR, FitTrace, quick = True, arc = False, nosub = True )
        
        spec     = spec[:,::-1]
        sig_spec = sig_spec[:,::-1]
        
        pickle.dump( spec, open( Conf.rdir + 'extracted_spec.pkl', 'wb' ) )
        pickle.dump( sig_spec, open( Conf.rdir + 'extracted_sigspec.pkl', 'wb' ) )
#        pickle.dump( spec, open( Conf.rdir + 'extracted_spec_quick.pkl', 'wb' ) )
#        pickle.dump( sig_spec, open( Conf.rdir + 'extracted_sigspec_quick.pkl', 'wb' ) )
            
    arcwavsol = Fns.Get_WavSol( wspec, sig_wspec, Conf )
    objwavsol = Fns.Interpolate_Obj_WavSol( arcwavsol, FileInfo, ArcInds, ObjInds, Conf )
  
    if Conf.ContDone:
        cont       = pickle.load( open( Conf.rdir + 'cont.pkl', 'rb' ) )
        spec_cf    = pickle.load( open( Conf.rdir + 'contfitspec.pkl', 'rb' ) )
        sigspec_cf = pickle.load( open( Conf.rdir + 'contfitsigspec.pkl', 'rb' ) )
    else:
        print( 'Fitting the continuum' )
        cont, spec_cf, sigspec_cf = Fns.getContinuum( spec, sig_spec, Conf )
    