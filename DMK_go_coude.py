import glob, os, pdb, readcol, pickle, mpyfit

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.optimize as optim
import scipy.interpolate as interp
import pandas as pd

from astropy.io import fits
from scipy import signal
from mpfit import mpfit

def Header_Info( dir, outname ):

    os.chdir(dir)
    files = glob.glob( '*.fits' )

    outfile = open( outname, 'wb' )
    heading = 'File,Object,RA,DEC,Type,ExpTime,Order,Airmass,UTdate,UT,gain,rdn,zenith\n'
    outfile.write(heading)
    
    for i in range( len( files ) ):
        hdulist    = fits.open(files[i])
        head       = hdulist[0].header
        itype      = head["imagetyp"]
        if len(files[i].split('.')) > 2: itype = "unknown"
        ra = ''
        dec = ''
        if "RA"  in head.keys(): ra      = head["RA"]
        if "DEC" in head.keys(): dec     = head["DEC"]
        order   = head["order"]
        air = '100'
        if "airmass" in head.keys(): air     = str(head["airmass"])
        UT      = head["UT"]
        exptime = str(head["exptime"])
        if "GAIN3" in head.keys():
            gain = str(head["gain3"])
            rdn  = str(head["rdnoise3"])
        else:
            gain = str(head["gain2"])
            rdn  = str(head["rdnoise2"])
        object  = head["object"]
        utdate  = head["DATE-OBS"]
        zd = head["ZD"]
        line = files[i] +','+object + ','+ ra +','+ dec+','+ itype+','+exptime+','+order+','+ air+','+utdate+','+ UT +','+gain+','+rdn+','+zd+' \n' 
        outfile.write(line)
        outfile.flush()

    outfile.close()
    return None

def Build_Bias( files ):

    testframe = fits.open( files[0] )
    testdata  = testframe[0].data
    biascube  = np.zeros( ( len( files ), testdata.shape[0], testdata.shape[1] ) )

    for i in range( len( files ) ):
        biascube[i] = fits.open( files[i] )[0].data

    SuperBias = np.median( biascube, axis = 0 )

    return SuperBias

def Build_Flat_Field( files, SuperBias ):

    testframe = fits.open( files[0] )
    testdata  = testframe[0].data
    flatcube  = np.zeros( ( len( files ), testdata.shape[0], testdata.shape[1] ) )

    for i in range( len( files ) ):
        flatcube[i] = fits.open( files[i] )[0].data - SuperBias

    FlatField  = np.median( flatcube, axis = 0 )
    FlatField -= np.min( FlatField )
    FlatField /= np.max( FlatField )

    return FlatField
    
def Basic_Cals( BiasFiles, FlatFiles, Conf ):

    if Conf.CalsDone == False:
        # Create master bias
        print 'Reading Bias Files'
        SuperBias = Build_Bias( BiasFiles )
        pickle.dump( SuperBias, open( Conf.rdir + 'bias.pkl', 'wb' ) )

        # Create master flat
        print 'Reading Flat Files'
        FlatField = Build_Flat_Field( FlatFiles, SuperBias )
        pickle.dump( FlatField, open( Conf.rdir + 'flat.pkl', 'wb' ) )

    elif Conf.CalsDone == True:
        print 'Reading in premade Bias and Flat files'
        SuperBias  = pickle.load( open( Conf.rdir + 'bias.pkl', 'rb' ) )
        FlatField  = pickle.load( open( Conf.rdir + 'flat.pkl', 'rb' ) )

    if Conf.PlotsOn:
        print 'Plotting bias:'
        plt.imshow( np.log10( SuperBias ), cmap = plt.get_cmap('gray'), aspect = 'auto', interpolation = 'none' )
        plt.colorbar(); plt.savefig( Conf.rdir + 'plots/bias.pdf' ); plt.show()

        print 'Plotting flat:'
        plt.imshow( np.log10( FlatField ), cmap = plt.get_cmap('gray'), aspect = 'auto', interpolation = 'none' )
        plt.colorbar(); plt.savefig( Conf.rdir + 'plots/flat.pdf' ); plt.show()
        
    return SuperBias, FlatField

def Make_BPM( Bias, Flat, CutLevel, Conf ):

    cutbias = np.percentile( Bias, CutLevel )
    BPM     = np.where( ( Bias > cutbias ) | ( Flat <= 0.0001 ) )

    if Conf.PlotsOn:
        plt.imshow( np.log10( Bias ), aspect = 'auto', interpolation = 'none' )
        plt.plot( BPM[1], BPM[0], 'r,' ) # Invert x,y for imshow
        print 'Plotting the bad pixel mask over the bias:'
        plt.savefig( Conf.rdir + 'plots/bpm.pdf' ); plt.show()

    return BPM

def Make_Cube( Files, ReadNoise, DarkCur, Bias = None, Flat = None, BPM = None ):

    for i in range( len( Files ) ):
        frame = fits.open( Files[i] )[0].data

        if i == 0:
            Cube = np.zeros( ( len( Files ), frame.shape[0], frame.shape[1] ) )
            SNR  = np.zeros( ( len( Files ), frame.shape[0], frame.shape[1] ) )

        Cube[i] = frame - DarkCur[0]
        SNR[i]  = Cube[i] / np.sqrt( Cube[i] + DarkCur[0] + ReadNoise[0] ** 2.0 )

        if Bias != None: Cube[i] -= Bias
        if Flat != None: Cube[i] /= Flat
        if BPM  != None:
            Cube[i, BPM[0], BPM[1]] = np.median( Cube[i] )
            SNR[i, BPM[0], BPM[1]]  = 0.001 # Effectively 0

        wherenans = np.where( np.isnan( Cube[i] ) )
        Cube[i, wherenans[0], wherenans[1]] = 1.0
        SNR[i, wherenans[0], wherenans[1]]  = 0.001

    return Cube, SNR

def Return_Cubes( ArcInds, ObjInds, FileInfo, DarkCube, Bias, Flat, BPM ):

    ReadNoise       = FileInfo.rdn[ArcInds] / FileInfo.gain[ArcInds]
    DarkCur         = DarkCube[ArcInds] / FileInfo.gain[ArcInds]
    ArcCube, ArcSNR = Make_Cube( FileInfo.File[ArcInds].values, ReadNoise.values, DarkCur.values, Bias = Bias )

    ReadNoise       = FileInfo.rdn[ObjInds] / FileInfo.gain[ObjInds]
    DarkCur         = DarkCube[ObjInds] / FileInfo.gain[ObjInds]
    ObjCube, ObjSNR = Make_Cube( FileInfo.File[ObjInds].values, ReadNoise.values, DarkCur.values, Bias = Bias, Flat = Flat, BPM = BPM )

    return ArcCube, ArcSNR, ObjCube, ObjSNR
        
def Start_Trace( flatslice, percent ):

    fgrad    = np.gradient( flatslice )
    cutvalue = np.percentile( abs( fgrad ), percent )

    orderzeros = []
    last       = 0

    for i in range( 6, flatslice.shape[0] ):
        if fgrad[i] > cutvalue or last == 0:
            if 100 > i - last > 20 or last == 0:
                orderzeros.append( i + 11 )
                last = i

    orderzeros = np.array( orderzeros )

    for i in range( len( orderzeros ) ):
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

def Find_Orders( Flat, orderstart ):

    midpoint = ( Flat.shape[1] + orderstart ) / 2

    startzeros, startvals = Start_Trace( Flat[:,orderstart], 60.0 )
    midzeros, midvals     = Start_Trace( Flat[:,midpoint], 45.0 )

    midzeros = midzeros[2:]
    midvals  = midvals[2:]

    slopes = []
    dx     = Flat.shape[1] + orderstart - midpoint
    for i in range( 30 ):
        dy = float( startzeros[i] - midzeros[i] )
        slopes.append( dy / dx )

    slopefit = np.polyfit( range( 30 ), slopes, 2 )

    finalzeros = np.round( midzeros + np.polyval( slopefit, range( len( midzeros ) ) ) * dx ).astype( int )
    finalvals  = Flat[finalzeros, orderstart]

    return finalzeros, finalvals


def Full_Trace( Cube, orderzeros, orderstart ):

    numord = len( orderzeros )
    trace  = np.zeros( ( Cube.shape[0], numord, Cube.shape[2] + orderstart ) )

    for i in range( Cube.shape[0] ):
        for pix in range( 1, Cube.shape[2] + orderstart + 1 ):
            prev = orderzeros
            if pix > 1: prev = trace[i,:,-pix+1]
            m1d = Cube[i,:,-pix+orderstart]
            for order in range( numord ):
                edge1 = int(prev[order] - 3)
                if edge1 < 0: edge1 = 0
                trace[i,order,-pix] = edge1 + np.argmax( m1d[edge1:edge1+6] )
                if pix != 1:
                    if ( trace[i,order,-pix] > prev[order] + 2 ) or ( trace[i,order,-pix] < prev[order] - 2 ):
                    #if prev[order] - 2 > trace[i,order,-pix] > prev[order] + 2:
                        trace[i,order,-pix] = prev[order]

    return trace

def Fit_Trace( Trace ):

    FitTrace = np.zeros( ( Trace.shape[0], 2048 ) )
    
    for order in range( Trace.shape[0] ):

        poly            = np.polyfit( np.arange( Trace.shape[1] ), Trace[order,:], 3 )
        vals            = np.polyval( poly, np.arange( 2048 ) )
        FitTrace[order] = vals

    return FitTrace

def Get_Trace( Flat, Cube, Conf ):

    orderstart            = -33
    orderzeros, ordervals = Find_Orders( Flat, orderstart )
    
    if Conf.TraceDone == False:
        print 'Performing preliminary trace'
        if Conf.PlotsOn:
            plt.plot( Flat[:,orderstart], 'k-' )
            plt.plot( orderzeros, ordervals, 'ro' )
            plt.savefig( Conf.rdir + 'plots/prelimtrace.pdf' ); plt.show()

        meds      = [ np.median( Cube[i,:,:2048] ) for i in range( Cube.shape[0] ) ]
        abovemed  = Cube[ np.where( meds >= np.percentile( meds, Conf.MedCut ) ) ]

        trace    = Full_Trace( abovemed, orderzeros, orderstart )
        MedTrace = np.median( trace, axis = 0 )
        FitTrace = Fit_Trace( MedTrace )
        if FitTrace[0,-1] < 8.0:
            MedTrace = MedTrace[1:]
            FitTrace = FitTrace[1:]
        print 'Saving median trace to file'
        pickle.dump( MedTrace, open( Conf.rdir + 'median_trace.pkl', 'wb' ) )
        pickle.dump( FitTrace, open( Conf.rdir + 'fitted_trace.pkl', 'wb' ) )

    elif Conf.TraceDone == True:
        print 'Reading in premade Trace and plotting on Flat:'
        MedTrace = pickle.load( open( Conf.rdir + 'median_trace.pkl', 'rb' ) )
        FitTrace = pickle.load( open( Conf.rdir + 'fitted_trace.pkl', 'rb' ) )

    if Conf.PlotsOn:
        plt.imshow( np.log10( Flat ), aspect = 'auto', cmap = plt.get_cmap( 'gray' ) )
        for i in range( FitTrace.shape[0] ):
            plt.plot( FitTrace[i,:], 'r-' )
        plt.xlim( 0, 2048 )
        plt.ylim( 2048, 0 )
        plt.savefig( Conf.rdir + 'plots/trace.pdf' ); plt.show()

    return MedTrace, FitTrace

def Least( p, args ):
    X, vals, err, func = args
    if err != None:
        dif = ( vals - func( X, p ) ) / err
    else:
        dif = vals - func( X, p )

    return dif.ravel()

def OrderModel( X, p, return_full = False ):
    x, y = X
    
    ##order trace residual (parabola)
    means  = p[2] * x ** 2.0 + p[1] * x + p[0]
    
    ##peak shape curve
    peaks  = p[5] * x ** 2.0 + p[4] * x + p[3]
    
    ##sigma curve
    sigmas = p[9] * x ** 3.0 + p[8] * x ** 2.0 + p[7] * x + p[6]
    
    ##actual model
    model  = peaks * np.exp( - ( y - means ) ** 2.0 / ( 2.0 * sigmas ** 2.0 ) ) + p[10]
    
    if return_full == False: return model
    else: return model, means, peaks, sigmas

def GaussModel( X, p ):
    x = X

    model = p[0] * np.exp( - ( x - p[1] ) ** 2.0 / ( 2.0 * p[2] ** 2.0 ) ) + p[3]

    return model

def Extractor( cube, cube_snr, trace, quick = True, arc = False, nosub = True ):

    flux  = np.zeros( ( cube.shape[0], trace.shape[0], trace.shape[1] ) )
    error = flux * 0.0

    for frm in range( cube.shape[0] ):
        print "Extracting Frame " + str(frm+1) +" out of " + str(cube.shape[0])
        thisfrm = cube[frm,:,:]
        thissnr = cube_snr[frm,:,:]
        
        for ord in range( trace.shape[0] ):

            tblock = np.zeros( ( trace.shape[1], 16 ) )
            tsnr   = tblock.copy()
            x, y   = [ c.T for c in np.meshgrid( np.arange( tblock.shape[0] ), np.arange( tblock.shape[1] ) ) ]

            for pix in range( trace.shape[1] ):
                low           = int(trace[ord,pix]) - 8
                high          = int(trace[ord,pix]) + 8
                tblock[pix,:] = thisfrm[low:high,pix]
                tsnr[pix,:]   = thissnr[low:high,pix]

            if (quick == False) & (arc == False):   

                ##clean obvious high outliers 
                toohigh         = np.where( tblock > 10.0 * np.median( tblock ) )
                tblock[toohigh] = np.median( tblock )
                tsnr[toohigh]   = 0.000001
                tnoise          = np.absolute( tblock / ( tsnr ) )

                ##clean zero values (often due to chip artifacts that aren't caught)
                toolow         = np.where( tblock <= 0 )
                tblock[toolow] = np.median( tblock )
                tsnr[toolow]   = 0.00001
                tnoise[toolow] = np.median( tblock ) * 2.0

                tnoise[tnoise==0.0] = 0.00001

                p0             = np.zeros( 11 )
                p0[[0,3,6,10]] = [ np.argmax( tblock[0,:] ), np.median( tblock[:,np.argmax( tblock[1000,:] )] ), 2.0, np.percentile( tblock, 2.0 ) ]

                ordpars, ordres = mpyfit.fit( Least, p0, ( (x,y), tblock, tnoise, OrderModel ) )

                if ordres['status'] == -16:
                    print 'mpyfit for this order failed?'
                    pdb.set_trace()

                bestmod, besttrace, bestpeak, bestsigmas = OrderModel( (x,y), ordpars, return_full = True )
                bestresid                                = ( tblock - bestmod ) / tnoise

                ##now remove points that are obvious outliers from the best model
                cutresid       = np.percentile( bestresid.flatten() , 99.99 )
                badcut         = np.where( bestresid > cutresid )
                tblock[badcut] = np.median( tblock )
                tsnr[badcut]   = 0.00001
                thistrace      = besttrace[:,0].copy()
                thispeak       = bestpeak[:,0].copy()
                thissigma      = bestsigmas[:,0].copy()
                peakshape      = np.sum( bestmod, axis = 1 )
                block_bg       = ordpars[5]
                block_sigma    = ordpars[4]

            print "Extracting Ord " + str(ord+1) +" out of " + str(trace.shape[0]) + " for frame " + str(frm+1) + " of " + str(cube.shape[0])

            savesigma = np.zeros(trace.shape[1])
            savepeak  = np.zeros(trace.shape[1])
            savebg    = np.zeros(trace.shape[1])
            savespot  = np.zeros(trace.shape[1])
            
            #go pixel-by-pixel along the order
            for pix in range(trace.shape[1]):
                slice     = tblock[pix,:]
                slice_snr = tsnr[pix,:]
                thisx     = np.arange(len(slice))
                snoise    = np.absolute(slice/slice_snr)
                snoise[snoise==0.0] = 0.00001
                qwe       = np.where(slice_snr < 0.0)[0]
                if len(qwe) >= 1: 
                    print 'Slice SNR is bad, bugshoot!!!'
                    pdb.set_trace()
                
                ##decide which extraction we do, e.g. a fast one, a detailed fit, or just get arc lines
                if quick == False: ##the detailed fit case
                    tspot     = thistrace[pix]
                    tpeak     = thispeak[pix]
                    tsigma    = thissigma[pix]

                    ##set up an mpfit of the order profile
                    ##initial parameters: peak height, peak centroid and peak width initalized from global order fit above, 
                    p0        = np.array( [ tpeak, tspot, tsigma, block_bg ] )
                    parinfo   = [ { 'limits': ( None, None ) } for i in range( len(p0) ) ]
                    parinfo[1]['limits'] = ( p0[1] - 1.0, p0[1] + 1.0 )
                    parinfo[2]['limits'] = ( p0[2] - 1.0, p0[2] + 2.0 )

                    slicepars, sliceres  = mpyfit.fit( Least, p0, ( thisx, slice, snoise, GaussModel ), parinfo = parinfo )

                    savesigma[pix] = slicepars[2]
                    savespot[pix]  = slicepars[1]
                    savepeak[pix]  = slicepars[0]
                    savebg[pix]    = slicepars[3]
                    bg     = slicepars[3]

                    #best model and residuals
                    themod = GaussModel( thisx, slicepars )
                    resids = ( slice - themod ) / snoise

                    ##detect outlier where residuals are really large
                    ##the cut value might need more thought 
                    bads   = np.where( np.absolute( resids ) > 20 )[0]
                    slice_snr[bads] = 0.0001 ##set their SNR to effective zero
                    if len(bads) > 3: slice_snr = slice_snr*0.000+0.0001 ##if there are three bad points of more, kill the whole pixel position

                ## if quick is true, simple minimum background
                if quick   == True: bg = slice * 0.0 + np.min( np.absolute( slice ) )
                if nosub   == True: bg = slice * 0.0 ##case for no subtraction at all (e.g., arcs)
                cslice     = slice - bg ## subtract the background, whatever the case was

                ##sum up across the order, just a SNR weighted mean
                flux[frm,ord,pix]  = np.sum(slice_snr*cslice)/np.sum(slice_snr)

                ##if this is an arc frame, don't do anything fancy for calculating the spectrum
                if arc == True: flux[frm,ord,pix] = np.sum(cslice)
                
                ##the uncertainty
                final_sig = np.sqrt(np.sum(slice_snr**2*snoise**2)/np.sum(slice_snr)**2)
                final_snr = np.absolute(flux[frm,ord,pix]/final_sig)

                if final_snr <0: 
                    print 'something has gone wrong, negative errors!!??!!'
                    pdb.set_trace()            
    
                error[frm,ord,pix] = final_sig*1.0
    
                ##clean bad points from chip artifacts
                badlow = np.where(flux[frm,ord] <=1)
                error[frm,ord,badlow] = np.median(flux[frm,ord])

    return flux,error

########## WAVELENGTH CALIBRATION FUNCTIONS AND WHAT NOT ##########

def Smooth_Spec( spec, specsig ):
    smoothed  = spec.copy()
    normfilt  = spec.copy()
    smoothsig = specsig.copy()
    
    for i in range( spec.shape[0] ):
        for j in range( spec.shape[1] ):
            cutspec         = spec[i,j].copy()
            cut             = spec[i,j] >= np.percentile( spec[i,j], 90.0 )
            cutspec[cut]    = np.percentile( spec[i,j], 75.0 )
            filtered        = signal.savgol_filter( cutspec, 101, 3 )
            smoothed[i,j]  /= filtered
            smoothsig[i,j] /= filtered
            normfilt[i,j]   = filtered / np.max( filtered )

    return smoothed, smoothsig, normfilt

def Get_Shift( cube, comp ):
    
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

def Get_WavSol( Cube, CubeSig, Conf, plots = True, Frames = 'All', Orders = 'All' ):

    if Conf.ArcWavDone:
        FullWavSol = pickle.load( open( Conf.rdir + 'wavsol.pkl', 'rb' ) )

    else:
        if not os.path.exists( Conf.rdir + 'wavcal' ):
            os.mkdir( Conf.rdir + 'wavcal' )

        if Frames == 'All': frameloop = range( Cube.shape[0] )
        else:               frameloop = Frames

        if Orders == 'All': orderloop = range( Cube.shape[1] )
        else:               orderloop = Orders

        roughsol = pickle.load( open( Conf.codedir + 'prelim_wsol.pkl', 'rb' ) )
        compspec = pickle.load( open( Conf.codedir + 'normfilt.pkl', 'rb' ) )

        smoothcube, smoothsig, filtcube = Smooth_Spec( Cube, CubeSig )

        orderdif = Get_Shift( filtcube[0], compspec )

        if orderdif < 0 or orderdif + Cube.shape[0] > roughsol.shape[0]:
            raw_input( 'Problem with number of orders found.\n' )

        FullWavSol  = np.zeros( ( Cube.shape[0], Cube.shape[1], Cube.shape[2] ) )
        FullParams  = np.zeros( ( Cube.shape[0], Cube.shape[1], 5 ) )
    
        THAR            = { 'wav': 0, 'spec': 0, 'logspec': 0, 'lines': 0 }
        THARcalib       = fits.open( Conf.codedir + 'thar_photron.fits' )[0]
        header          = THARcalib.header
        THAR['spec']    = THARcalib.data
        THAR['wav']     = np.arange( len(THAR['spec']) ) * header['CDELT1'] + header['CRVAL1']
        THAR['logspec'] = np.log10( THAR['spec'] )
        THAR['lines']   = pd.read_table( Conf.codedir + 'ThAr_list.txt', delim_whitespace = True ).wav.values

        badorders = [ [] for i in range( len(frameloop) ) ]
        
        i = 0
        for frame in frameloop:
            framepath = Conf.rdir + 'wavcal/arcframe_' + str( frame )
            if not os.path.exists( framepath ):
                os.mkdir( framepath )

            for order in orderloop:
                orderpath = framepath + '/order_' + str(order)
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
                #logarcspec = np.log10( arcspec - np.min( arcspec ) + 1.0 )
                #logarcspec = logarcspec - np.min( logarcspec )

                wavsol, params, keeps, rejs, flag = Fit_WavSol( prelimsol, arcspec, arcsigma, THAR['lines'], orderpath, THAR, plots = plots )
            
                if flag:
                    badorders[i].append(order)
                
                pickle.dump( keeps, open( orderpath + '/peakskept_ord_' + str(order) + '.pkl', 'wb' ) )
                pickle.dump( rejs, open( orderpath + '/peaksrej_ord_' + str(order) + '.pkl', 'wb' ) )
            
                if plots:
                    plt.clf()
                    plt.plot( wavsol, logarcspec, 'k-', lw = 1 )
                    plt.plot( THAR['wav'], THAR['logspec'], 'r-', lw = 1 )
                    plt.xlim( wavsol[0], wavsol[-1] )
                    for peak in keeps['line']:
                        plt.axvline( x = peak, color = 'b', ls = ':', lw = 1 )
                    plt.savefig(orderpath + '/fullspec_ord_' + str(order) + '.pdf' )

                    Plot_Wavsol_Windows( wavsol, keeps['line'], logarcspec, THAR, orderpath, frame, order )

                FullWavSol[frame,order] = wavsol
                FullParams[frame,order] = params

            i += 1

    pickle.dump( FullWavSol, open( Conf.rdir + 'wavsol.pkl', 'wb' ) )
    pickle.dump( FullParams, open( Conf.rdir + 'wavparams.pkl', 'wb' ) )
    
    print badorders
    return FullWavSol

def Find_Peaks( wav, spec, specsig, peaksnr = 5, pwidth = 10, minsep = 0.5 ):
        
    # Find peaks using the cwt routine from scipy.signal
    peaks = signal.find_peaks_cwt( spec, np.arange( 2, 4 ), min_snr = peaksnr, noise_perc = 25.0 )
    
    # Offset from start/end of spectrum by some number of pixels
    peaks = peaks[ (peaks > pwidth) & (peaks < len(spec) - pwidth) ]
    
    pixcent = np.array([])
    wavcent = np.array([])
        
    for peak in peaks:
        
        xi   = wav[peak - pwidth:peak + pwidth]
        yi   = spec[peak - pwidth:peak + pwidth]
        sigi = specsig[peak - pwidth:peak + pwidth]
        inds = np.arange( len(xi), dtype = float )
        
        p0       = [ yi[9], np.median( inds ), 0.9, np.median( spec ) ]
        lowerbds = [ p0[3], p0[1] - 2.0, 0.3, 0.0  ]
        upperbds = [ None, p0[1] + 2.0, 1.5, None ]
        parinfo  = [ { 'limits': ( lowerbds[i], upperbds[i] ) } for i in range( len(p0) ) ]

        try:
            params, res = mpyfit.fit( Least, p0, ( inds, yi, sigi, GaussModel ), parinfo = parinfo )

            pixval  = peak - pwidth + params[1]
            pixcent = np.append( pixcent, pixval )
            
            ceiling = np.ceil( pixval ).astype(int)
            floor   = np.floor( pixval ).astype(int)
            slope   = ( wav[ceiling] - wav[floor] ) / ( ceiling - floor )
            wavval  = wav[floor] + slope * ( pixval - floor )
            wavcent = np.append( wavcent, wavval )

            # fitx = np.linspace( inds.min(), inds.max(), 1000 )
            # plt.step( inds, yi, 'k', where = 'mid' )
            # plt.plot( fitx, GaussModel(fitx,params), 'r-' )
            # plt.errorbar( inds, yi, yerr = sigi, fmt = 'none' )
            # plt.title( str(peak) )
            # plt.show()

        except RuntimeError:
            print 'RuntimeError'
            #pdb.set_trace()
    
    vals = spec[pixcent.astype(int)]
    oks  = np.ones( len(pixcent), int )
    
    for i in range( len(wavcent) ):
        dist  = np.absolute( wavcent - wavcent[i] )
        close = np.where( dist <= minsep )[0]
        small = np.where( vals[close] < np.max( vals[close] ) )[0]
        if len(small) != 0: oks[close[small]] = -1
            
    keep    = np.where( oks == 1 )
    pixcent = pixcent[keep]
    wavcent = wavcent[keep]

    return pixcent, wavcent

def Fit_WavSol( wav, spec, specsig, THARcat, path, THAR, snr = 5, minsep = 0.5, plots = True ):
    
    wavsol = np.zeros( len(spec) )
    
    pixcent, wavcent = Find_Peaks( wav, spec, specsig, peaksnr = snr, minsep = minsep )
    
    keeps = { 'pix': np.array([]), 'wav': np.array([]), 'line': np.array([]) }
    rejs  = { 'pix': np.array([]), 'wav': np.array([]), 'line': np.array([]) }
    
    for i in range( len(wavcent) ):
        dists    = np.absolute( THARcat - wavcent[i] )
        mindist  = np.argmin( dists )
            
        if dists[mindist] <= 1.0:
            keeps['pix']  = np.append( keeps['pix'], pixcent[i] )
            keeps['wav']  = np.append( keeps['wav'], wavcent[i] )
            keeps['line'] = np.append( keeps['line'], THARcat[mindist] )

    dofit  = True
    ploti  = 1
    cutoff = 3.0 / 0.67449 # Corrects MAD to become sigma
    
    while dofit:
        wavparams  = np.polyfit( keeps['pix'], keeps['line'], 4 )
        ptsfromfit = np.polyval( wavparams, keeps['pix'] )
        
        wavresids  = ptsfromfit - keeps['line']
        velresids  = wavresids / keeps['line'] * 3e5
        resids     = { 'wav': wavresids, 'vel': velresids }
        medabsdev  = np.median( np.abs( np.abs( resids['wav'] ) - np.median( np.abs( resids['wav'] ) ) ) )
        
        velcut = np.sum( np.abs(resids['vel']) >= 0.2 )
        torej  = np.abs( resids['wav'] ) >= cutoff * medabsdev
        tokeep = np.logical_not( torej )
        numrej = np.sum( torej )
        
        # velcut     = np.sum( np.abs(resids['vel']) >= 0.2 )
        # gtcut      = np.abs( resids['wav'] ) >= cutoff * np.median( np.abs( resids['wav'] ) )
        # numrej     = np.sum( gtcut )
        # big        = np.argmax( np.abs( resids['wav'] ) )
        # torej      = np.zeros( len(resids['wav']), dtype = np.bool )
        # torej[big] = True
        # tokeep     = np.logical_not( torej )
        
        if velcut > 0 and numrej != len(torej):
            if numrej > 0:
                if plots:
                    plotname = path + '/resids_round_' + str(ploti) + '.pdf'
                    Plot_WavSol_Resids( resids, keeps['line'], cutoff, plotname, tokeep = tokeep, toreject = torej )

                    pltwav = np.polyval( wavparams, np.arange( len( spec ) ) )
                    plt.clf()
                    plt.plot( pltwav, np.log10(spec), 'k-', lw = 1 )
                    plt.plot( THAR['wav'], THAR['logspec'], 'r-', lw = 1 )
                    plt.xlim( pltwav[0], pltwav[-1] )
                    for peak in keeps['line']:
                        plt.axvline( x = peak, color = 'b', ls = ':', lw = 1 )
                    plt.savefig(path + '/rejplots/fullspec_' + str(ploti) + '.pdf')

                    # wav = np.polyval(wavparams,np.arange(len(spec)))
                    # plt.clf()
                    # plt.plot( wav, np.log10(spec-spec.min()), 'k-', lw = 1 )
                    # plt.plot( THAR['wav'], THAR['logspec'], 'r-', lw = 1 )
                    # plt.xlim( wav[0], wav[-1] )
                    # for peak in keeps['line'][tokeep]:
                    #     plt.axvline( x = peak, color = 'b', ls = ':', lw = 1 )
                    # for peak in keeps['line'][torej]:
                    #     plt.axvline( x = peak, color = 'g', ls = ':', lw = 1 )
                    # plt.show()

                    # Plot_Wavsol_Windows( wav, keeps['line'][tokeep], np.log10(spec-spec.min()), THAR, None, 0, 0 )
                
                rejs['pix']  = keeps['pix'][torej]
                rejs['wav']  = keeps['wav'][torej]
                rejs['line'] = keeps['line'][torej]
                
                keeps['pix']  = keeps['pix'][tokeep]
                keeps['wav']  = keeps['wav'][tokeep]
                keeps['line'] = keeps['line'][tokeep]
                
                ploti += 1
                
            elif numrej == 0 and cutoff == 3.0 / 0.67449:
                cutoff = 2.0 / 0.67449
                
            else:
                print 'There is something seriously wrong.\n'
                print 'There are points > 0.2 km/s, but none are found to be rejected. FIX'
                flag = True
                if plots:
                    plotname = path + '/resids_round_' + str(ploti) + '_flag.pdf'
                    Plot_WavSol_Resids( resids, keeps['line'], cutoff, plotname )
                break

        elif numrej == len(torej):
            if plots:
                plotname = path + '/resids_round_' + str(ploti) + '.pdf'
                Plot_WavSol_Resids( resids, keeps['line'], cutoff, plotname, toreject = torej )
            flag = True
            dofit = False            

        else:
            if plots:
                plotname = path + '/resids_round_' + str(ploti) + '.pdf'
                Plot_WavSol_Resids( resids, keeps['line'], cutoff, plotname, tokeep = tokeep )
            flag = False
            dofit = False
            
    wavsol = np.polyval( wavparams, np.arange( len(spec) ) )

    return wavsol, wavparams, keeps, rejs, flag

def Plot_WavSol_Resids( resids, lines, cutoff, savename, tokeep = None, toreject = None ):

    wavmad = np.median( np.abs( np.abs( resids['wav'] ) - np.median( np.abs( resids['wav'] ) ) ) )
    velmad = np.median( np.abs( np.abs( resids['vel'] ) - np.median( np.abs( resids['vel'] ) ) ) )
    
    plt.clf()
    fig, (wavax, velax) = plt.subplots( 2, 1, sharex = 'all' )
    wavax.axhline( y = 0.0, color = 'k', ls = ':' )
    velax.axhline( y = 0.0, color = 'k', ls = ':' )
    if toreject == None:
        wavax.plot( lines, resids['wav'], 'ko', mfc = 'none' )
        velax.plot( lines, resids['vel'], 'ko', mfc = 'none' )
        fig.suptitle( 'Lines Used: ' + str(lines.size) + ', Cutoff: ' + str(cutoff * 0.67449) + ' $\sigma$' )
    elif tokeep == None:
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
    plt.savefig(savename)
    ### REMOVE AFTER TESTING
    #plt.show()
    
    return None

def Plot_Wavsol_Windows( wavsol, wavkeep, spec, THAR, path, frame, order ):
    
    numfigs = np.ceil( np.ceil( np.ptp( wavsol ) / 10.0 ) / 6.0 ).astype(int)
    start   = np.min( wavsol )
    if numfigs > 4:
        return None
    else:
        for i in range(numfigs):
            fig   = plt.figure()
            j = 1
            while start <= np.max( wavsol ) and j < 7:
                sbplt = 230 + j
                fig.add_subplot(sbplt)
                plt.plot( wavsol, spec, 'k-', lw = 1 )
                plt.plot( THAR['wav'], THAR['logspec'], 'r-', lw = 1 )
                for peak in wavkeep:
                    plt.axvline( x = peak, color = 'b', ls = ':', lw = 1 )
                plt.xlim( start, start + 10.0 )
                plt.yticks( [], [] )
                low = np.round( start, 1 )
                plt.xticks( np.linspace( low, low + 10.0, 3 ), fontsize = 5 )
                start += 10.0
                j += 1
            plt.suptitle( 'Frame: ' + str(frame) + ', Order: ' + str(order) + ', Window: ' + str(i) )
            if path != None: plt.savefig( path + '/specwindow_' + str(i) + '.pdf' )
            else: plt.show()
    
        return None

########## WAVELENTH INTERPOLATION ##########

def Interpolate_Obj_WavSol( wavsol, info, arcinds, objinds, Conf ):

    if Conf.ObjWavDone:
        Wsol_Obj = pickle.load( open( Conf.rdir + 'objwavsol.pkl', 'rb' ) )

    else:
        juldays  = UT_Convert( info.UT.values, info.UTdate.values )
        arctime  = juldays[arcinds]
        objtime  = juldays[objinds]

        Wsol_Obj = np.zeros( ( len(objtime), wavsol.shape[1], wavsol.shape[2] ) )

        for i in range( len( objtime ) ):
            deltime = objtime[i] - arctime

            if np.max( arctime ) < objtime[i] or np.min( arctime ) > objtime[i]:
                print 'Target obs ' + str(i+1) + ' not bounded by arcs, using closest arc.'
                closest     = np.argmin( np.abs( deltime ) )
                Wsol_Obj[i] = wavsol[closest]
            else:
                bef = np.where( deltime > 0 )[0][-1]
                aft = np.where( deltime < 0 )[0][0]
                Wsol_Obj[i] = wavsol[bef] + ( wavsol[aft] - wavsol[bef] ) / ( arctime[aft] - arctime[bef] ) * ( objtime[i] - arctime[bef] )

        pickle.dump( Wsol_Obj, open( Conf.rdir + 'objwavsol.pkl', 'wb' ) )

    return Wsol_Obj

def Date_To_JD( year, month, day ):
    '''
    calculate julian day from calendar day
    '''
    if month == 1 or month == 2:
        yearp  = year - 1
        monthp = month + 12
    else:
        yearp  = year
        monthp = month
    
    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or
        (year == 1582 and month < 10) or
        (year == 1582 and month == 10 and day < 15)):
        # before start of Gregorian calendar
        B = 0
    else:
        # after start of Gregorian calendar
        A = np.trunc(yearp / 100.)
        B = 2 - A + np.trunc(A / 4.)
        
    if yearp < 0:
        C = np.trunc((365.25 * yearp) - 0.75)
    else:
        C = np.trunc(365.25 * yearp)
        
    D  = np.trunc(30.6001 * (monthp + 1))
    
    jd = B + C + D + day + 1720994.5
    
    return jd  

def UT_Convert( UT, UTdate ):
    julday = np.zeros( len(UT) )
    
    for i in range( len(UT) ):
        h, m, s     = np.array( UT[i].split(':') ).astype( np.float64 )
        dayfrac     = h / 24.0 + m / 1440.0 + s / 86400.0
        yr, mth, dy = np.array( UTdate[i].split('-') ).astype( np.float64 )
        julday[i]   = Date_To_JD( yr, mth, dy + dayfrac )
        
    return julday
