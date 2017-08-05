import glob, os, pdb, readcol

import numpy as np
import matplotlib.pyplot as plt
import pickle

from astropy.io import fits
import scipy.optimize as optim
import scipy.interpolate as interp
from scipy import signal
from mpfit import mpfit

def header_info( dir, outname ):

    os.chdir(dir)
    files = glob.glob( '*.fits' )

    outfile = open( outname, 'wb' )
    heading = 'File,Object,RA,DEC,Type,ExpTime,Order,Airmass,UTdate,UT,gain,rdn,zenith \n'
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
        gain = str(head["gain3"])
        object  = head["object"]
        rdn     = str(head["rdnoise3"])
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
    biascube = np.zeros( ( len( files ), testdata.shape[0], testdata.shape[1] ) )

    for i in range( len( files ) ):
        biascube[i] = fits.open( files[i] )[0].data

    SuperBias = np.median( biascube, axis = 0 )

    return SuperBias

def Build_Flat_Field( files, SuperBias ):

    testframe = fits.open( files[0] )
    testdata  = testframe[0].data
    flatcube = np.zeros( ( len( files ), testdata.shape[0], testdata.shape[1] ) )

    for i in range( len( files ) ):
        flatcube[i] = fits.open( files[i] )[0].data - SuperBias

    FlatField = np.median( flatcube, axis = 0 )
    FlatField -= np.min( FlatField )
    FlatField /= np.max( FlatField )

    return FlatField
    
def Basic_Cals( BiasFiles, FlatFiles, CalsDone, rdir, plots = False ):

    if CalsDone == False:
        
        # Create master bias
        print 'Reading Bias Files'
        SuperBias = Build_Bias( BiasFiles )
        if plots == True:
            print 'Plotting bias:'
            plt.imshow( np.log10( SuperBias ), cmap = plt.get_cmap('gray'), aspect = 'auto', interpolation = 'none' )
            plt.show()
        pickle.dump( SuperBias, open( rdir + 'bias.pkl', 'wb' ) )

        # Create master flat
        print 'Reading Flat Files'
        FlatField = Build_Flat_Field( FlatFiles, SuperBias )
        if plots == True:
            print 'Plotting flat:'
            plt.imshow( np.log10( FlatField ), cmap = plt.get_cmap('gray'), aspect = 'auto', interpolation = 'none' )
            plt.colorbar(); plt.show()
        pickle.dump( FlatField, open( rdir + 'flat.pkl', 'wb' ) )

    elif CalsDone == True:
        
        print 'Reading in premade Bias and Flat files'
        SuperBias  = pickle.load( open( rdir + 'bias.pkl', 'rb' ) )
        FlatField  = pickle.load( open( rdir + 'flat.pkl', 'rb' ) )

    return SuperBias, FlatField

def Make_BPM( Bias, Flat, CutLevel, ShowBPM ):

    cutbias = np.percentile( Bias, CutLevel )
    BPM     = np.where( ( Bias > cutbias ) | ( Flat <= 0.0001 ) )

    if ShowBPM:
        plt.imshow( np.log10( Bias ), aspect = 'auto', interpolation = 'none' )
        plt.plot( BPM[1], BPM[0], 'r,' ) # Invert x,y for imshow
        print 'Plotting the bad pixel mask over the bias:'
        plt.show()

    return BPM

def Make_Cube( Files, ReadNoise, DarkCur, Bias = None, Flat = None, BPM = None ):

    for i in range( len( Files ) ):
        #print 'Reading ' + files[i]
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

def Find_Orders( Flat, OrderStart ):

    midpoint = ( Flat.shape[1] + OrderStart ) / 2

    startzeros, startvals = Start_Trace( Flat[:,OrderStart], 60.0 )
    midzeros, midvals     = Start_Trace( Flat[:,midpoint], 45.0 )

    midzeros = midzeros[2:]
    midvals  = midvals[2:]

    slopes = []
    dx     = Flat.shape[1] + OrderStart - midpoint
    for i in range( 30 ):
        dy = float( startzeros[i] - midzeros[i] )
        slopes.append( dy / dx )

    slopefit = np.polyfit( range( 30 ), slopes, 2 )

    finalzeros = np.round( midzeros + np.polyval( slopefit, range( len( midzeros ) ) ) * dx ).astype( int )
    finalvals  = Flat[finalzeros, OrderStart]

    return finalzeros, finalvals


def Full_Trace( Cube, orderzeros, OrderStart ):

    numord = len( orderzeros )
    trace  = np.zeros( ( Cube.shape[0], numord, Cube.shape[2] + OrderStart ) )

    for i in range( Cube.shape[0] ):
        for pix in range( 1, Cube.shape[2] + OrderStart + 1 ):
            prev = orderzeros
            if pix > 1: prev = trace[i,:,-pix+1]
            m1d = Cube[i,:,-pix+OrderStart]
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

    FitTrace = Trace.copy()
    
    for order in range( Trace.shape[0] ):

        splfit          = interp.InterpolatedUnivariateSpline( range( Trace.shape[1] ), Trace[order,:] )
        vals            = splfit( range( Trace.shape[1] ) )
        vals[0]         = vals[1]
        FitTrace[order] = vals

    return FitTrace

def Get_Trace( Flat, Cube, OrderStart, MedCut, rdir, TraceDone, plots = False ):

    orderzeros, ordervals = Find_Orders( Flat, OrderStart )
    
    if TraceDone == False:
        print 'Performing preliminary trace'
        if plots == True:
            plt.plot( Flat[:,OrderStart], 'k-' )
            plt.plot( orderzeros, ordervals, 'ro' )
            plt.show()

        meds      = [ np.median( Cube[i,:,:2048] ) for i in range( Cube.shape[0] ) ]
        abovemed  = Cube[ np.where( meds >= np.percentile( meds, MedCut ) ) ]

        trace    = Full_Trace( abovemed, orderzeros, OrderStart )
        MedTrace = np.median( trace, axis = 0 )
        FitTrace = Fit_Trace( MedTrace )
        print 'Saving median trace to file'
        pickle.dump( MedTrace, open( rdir + 'median_trace.pkl', 'wb' ) )
        pickle.dump( FitTrace, open( rdir + 'fitted_trace.pkl', 'wb' ) )

    elif TraceDone == True:
        print 'Reading in premade Trace and plotting on Flat:'
        MedTrace = pickle.load( open( rdir + 'median_trace.pkl', 'rb' ) )
        FitTrace = pickle.load( open( rdir + 'fitted_trace.pkl', 'rb' ) )

    if plots == True:
        plt.imshow( np.log10( Flat ), aspect = 'auto', cmap = plt.get_cmap( 'gray' ) )
        for i in range( len( orderzeros ) ):
            plt.plot( FitTrace[i,:], 'r-' )
        plt.xlim( 0, 2048 )
        plt.ylim( 2048, 0 )
        plt.show()

    return MedTrace, FitTrace

def extractor(cube,cube_snr,trace,quick=True,arc=False,nosub=True):
    '''
    Main code to turn data images into a series 1d-spectra (one per order)
    Input:
        cube: the datacube of image frames
        cube_snr: corresponding SNR cube
        trace: The order position trace on the ccd, computed previously 
    
    Options:
        quick: if True, does a simple baground minimum subtractions and skips slow fitting process,
                good for quicklook or testing the pipelines other functions. If False, full fit is done.
        arc: set to True for extracting arc frames, should also set quick to True
    
        nosub: if True, no background subtraction carried out at all, mainly a diagnostic tool
    
    Outputs: 
        flux: cube (frames,orders,pixel) containing the extracted raw spectra
        error: corresponding uncertainties (same shape)
    
    General Overview of Steps:

    for each order in each data frame:
    


    '''
    flux  = np.zeros((cube.shape[0],trace.shape[0],trace.shape[1]))
    error = flux*0.0
    ##go frame by frame
    for frm in range(cube.shape[0]):
        #frm=-1 ##set a frame here for testing
        print "Extracting Frame " + str(frm+1) +" out of " + str(cube.shape[0])
        thisfrm = cube[frm,:,:]
        thissnr = cube_snr[frm,:,:]
        
        ## go order by order 
        for ord in range(trace.shape[0]):
            #ord=21 set an order here for testing

            ##fix the order trace shape first, to account for random pixel shifts 
            ###!!!DANNY this could really be moved to the trace functions as an additional step
            ##rather than doing it multiple times here!!!!!
            # tracepars = np.polyfit(range(trace.shape[1]),trace[ord,:],2)
            # tracepoly = np.polyval(tracepars,range(trace.shape[1]))
            # trace[ord] = tracepoly
            
            #now get the whole trace in a block
            tblock = np.zeros((trace.shape[1],16))
            tsnr   = tblock.copy()
            x      = np.arange(tblock.shape[0])
            y      = np.arange(tblock.shape[1])
            xx,yy  = np.meshgrid(x,y)
            xx     = xx.T
            yy     = yy.T
            ##cut out a region around the trace for this order
            for pix in range(trace.shape[1]):
                low  = int(trace[ord,pix]) - 8
                high = int(trace[ord,pix]) + 8
                if low < 0:
                    high -= low
                    low   = 0
                tblock[pix,:] = thisfrm[low:high,pix]
                tsnr[pix,:]   = thissnr[low:high,pix]
##a diagnostic plot for testing
#             plt.imshow(thisfrm[trace[ord,1350]-8:trace[ord,1350]+8,1330:1360],aspect='auto')
#             plt.plot(range(30),trace[ord,1330:1360]-trace[ord,1350]+9)
#             test = thisfrm[trace[ord,1330:1360]
#             pdb.set_trace()

            ##if not running the quick mode
            if (quick == False) & (arc == False):   

                ##clean obvious high outliers 
                trybad = np.where(tblock > 10*np.median(tblock))
                tsnr[trybad[0],trybad[1]] = 0.000001
                tblock[trybad[0],trybad[1]] = np.median(tblock)
                err = np.absolute(tblock/(tsnr))

                ##clean zero values (often due to chip artifacts that aren't caught)
                badpoints = np.where(tblock <= 0)
                tblock[badpoints[0],badpoints[1]] = np.median(tblock)
                err[badpoints[0],badpoints[1]] = np.median(tblock)*2.0
                tsnr[badpoints[0],badpoints[1]] = 0.00001                
                
                ##set up a fit of the order position on the trace block
                fa  = {'xx':xx, 'yy':yy, 'image':tblock,'err':err}
                ##initialize the normalizations paramaters to the centre values of tblock
                p0  = np.array([np.argmax(tblock[0,:]),0.0,0.0,np.median(tblock[:,np.argmax(tblock[1000,:])]),2.0,np.percentile(tblock,2),0.0,0.0,0.0,0.0,0.0]) 
                thisfit2d    = mpfit(order_resid, p0, functkw=fa,quiet=True) 
                if thisfit2d.status == -16: 
                    print 'mpfit for order fialed!, not sure what happened???!!!!???'
                    pdb.set_trace()
                ##the best fit parameters
                orderpars    = thisfit2d.params
                ##the corresponding best model  
                bestmod,besttrace,bestresid,bestpeak,bestsigmas    = order_resid(orderpars,xx=xx,yy=yy,image=tblock,err=tblock/tsnr,model=True)
                
                ##now remove points that are obvious outliers from the best model
                cutresid    = np.percentile(bestresid.flatten(),99.99)
                badcut      = np.where(bestresid > cutresid)
                tblock[badcut[0],badcut[1]] = np.median(tblock)
                tsnr[badcut[0],badcut[1]]   = 0.00001
                thistrace   = besttrace[:,0].copy()
                thispeak    = bestpeak[:,0].copy()
                thissigma   = bestsigmas[:,0].copy()
                peakshape   = np.sum(bestmod,axis=1)
                block_bg    = orderpars[5]
                block_sigma = orderpars[4]

            
            ## now that a starting order shape fit is done, can go pixel-by-pixel across the order
            ## and do a more detailed fit
            
            ##a quick print update so I don't get antsy            
            print "Extracting Ord " + str(ord+1) +" out of " + str(trace.shape[0]) + " for frame " + str(frm+1) + " of " + str(cube.shape[0])


            savesigma = np.zeros(trace.shape[1])
            savepeak  = np.zeros(trace.shape[1])
            savebg    = np.zeros(trace.shape[1])
            savespot  = np.zeros(trace.shape[1])
            #go pixel-by-pixel along the order
            for pix in range(trace.shape[1]):
                ##pix = 1199 #set a pixel here for testing
                slice     = tblock[pix,:]
                slice_snr = tsnr[pix,:]
                thisx     = np.arange(len(slice))
                snoise = np.absolute(slice/slice_snr)        
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
                    p0 = np.array([tpeak,tspot,tsigma,block_bg])  

                    ##limit some parameters, e.g. the width and centroid locations shoulndt travel far
                    ###this is mostly needed for the case of low-snr data where the peak is hard to pick out  
                    parinfo = [{'fixed':0, 'limited':[0,0], 'limits':[0.,0.]} for i in range(4)]
                    parinfo[1]['fixed']   = 0
                    parinfo[2]['fixed']   = 0
                    parinfo[1]['limited'] =[1,1]
                    parinfo[1]['limits']  =[p0[1]-1.0,p0[1]+1.0]
                    parinfo[2]['limited'] = [1,1]
                    parinfo[2]['limits']  = [p0[2]-1.0,p0[2]+2.0]
                    fa = {'x':thisx, 'y':slice, 'err':slice/slice_snr}

                    ##the call to the fitter
                    thisfit    = mpfit(profile_resid, p0, functkw=fa,quiet=True,parinfo=parinfo)    
                    #best fit parameters, save them
                    fitpars    = thisfit.params
                    savesigma[pix] = fitpars[2]
                    savespot[pix]  = fitpars[1]
                    savepeak[pix]  = fitpars[0]
                    savebg[pix]         = fitpars[3]
                    bg         = fitpars[3]

                    #best model and residuals
                    resids = profile_resid(fitpars,x=thisx,y=slice,err=slice/slice_snr,model=False,fjac=None)[1]
                    themod = profile_resid(fitpars,x=thisx,y=slice,err=slice/slice_snr,model=True,fjac=None)

                    ##detect outlier where residuals are really large
                    ##the cut value might need more thought 
                    bads = np.where(np.absolute(resids)>20)[0]
                    slice_snr[bads] = 0.0001 ##set their SNR to effective zero
                    if len(bads) > 3: slice_snr = slice_snr*0.000+0.0001 ##if there are three bad points of more, kill the while pixel position

                ## if quick is true, simple minimum background
                if quick   == True: bg = slice*0.0 + np.min(np.absolute(slice))
                if nosub   == True: bg = slice*0.0 ##case for no subtraction at all (e.g., arcs)
                cslice  = slice - bg ## subtract the background, whatever the case was

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

def profile_resid(p,x=None,y=None,err=None,model=False,fjac=None):
    '''
    mpfit style function of the order profile model, for doing the detailed extraction and 
    background removal

    The model is a gaussian with a constant level, 

    improvement idea: add a linear background across order? might not be needed

    if model=True, returns model to user, otherwise returns residuals for mpfit
    '''
    mmm = p[0]*np.exp(-(x-p[1])**2/2/p[2]**2)+ p[3]
    resid = (y-mmm)/err
    status=0
    #pdb.set_trace()
    if model == True: return mmm 
    if model == False: return [status,resid]

def order_resid(p,xx=None,yy=None,image=None,err=None,model=False,fjac=None):
    '''
    Function in the mpfit style to compute a model order shape 
    its a gaussian at each pixel row, with peak heights, positions, and widths varying parabolically

    p is the input parameters

    returns residuals for mpfit unless model=True is set, then it returns information for the user to eyeball things
    '''
    ##order trace residual (parabola)
    mvect  = p[2]*xx**2 + p[1]*xx + p[0]
    ##peak shape curve
    peak   = p[3] + p[6]*xx + p[7]*xx**2
    ##sigma curve
    sigmas = p[4]+p[8]*xx**1 + p[9]*xx**2 + p[10]*xx**3
    ##actual model
    mmm    = peak*np.exp(-(yy-mvect)**2/2/sigmas**2) + p[5]
    resid_2d = (image-mmm)/err
    resid    = resid_2d.flatten()
    status=0
    #pdb.set_trace()
    if model == True: return mmm,mvect,resid_2d,peak,sigmas
    if model == False: return [status,resid]

'''def Order_Model_2D( X, *p ):

    rav = True
    
    if not np.isscalar( p[0] ):
        p   = p[0]
        rav = False
    
    x, y = X
        
    mean  = p[2] * x ** 2.0 + p[1] * x + p[0]
    peak  = p[5] * x ** 2.0 + p[4] * x + p[3]
    sigma = p[9] * x ** 3.0 + p[8] * x ** 2.0 + p[7] * x+ p[6]

    model = peak * np.exp( - ( y - mean ) ** 2.0 / ( 2.0 * sigma ** 2.0 ) ) + p[10]

    if rav == False:
        return model, mean, peak, sigma
    else:
        return model.ravel()

def Order_Model_Profile( X, *p ):

    model = p[0] * np.exp( - ( X - p[1] ) ** 2.0 / ( 2.0 * p[2] ** 2.0 ) ) + p[3]

    return model

def Extraction( Cube, Cube_SNR, Trace, quick = True, arc = False, nosub = True ):

    if ExtractDone == True:
        Spec_Cube = pickle.load( open( rdir + 'extracted_spec.pkl', 'rb' ) )
        Spec_Sig  = pickle.load( open( rdir + 'extracted_specsig.pkl', 'rb' ) )

    elif ExtractDone == False:
        Flux  = np.zeros( ( Cube.shape[0], Trace.shape[0], Trace.shape[1] ) )
        Error = np.zeros( ( Cube.shape[0], Trace.shape[0], Trace.shape[1] ) )

        for frmi in range( Cube.shape[0] ):
            print "Extracting Frame # " + str( frame + 1 ) + ' out of ' + str( Cube.shape[0] )

            Frame      = Cube[ frmi, :, : ]
            FrameSNR   = Cube_SNR[ frmi, :, : ]

            for ord in range( Trace.shape[0] ):
                block = np.zeros( ( Trace.shape[1], 16 ) )
                snr   = block.copy()
                x, y  = [ c.T for c in np.meshgrid( np.arange( block.shape[0] ), np.arange( block.shape[1] ) ) ]

                for pix in range( Trace.shape[1] ):
                    block[pix,:] = Frame[np.round(Trace[ord,pix])-8:np.round(Trace[ord,pix])+8,pix]
                    snr[pix,:]   = FrameSNR[np.round(Trace[ord,pix])-8:np.round(Trace[ord,pix])+8,pix]

                if quick == False and arc == False:

                    # Get rid of obviously high outliers
                    toohigh        = np.where( block > 10.0 * np.median( block ) )
                    block[toohigh] = np.median( block )
                    snr[toohigh]   = 1.0e-6
                    err            = np.absolute( block / snr )

                    # Clean zero values
                    toolow        = np.where( block <= 0.0 )
                    block[toolow] = np.median( block )
                    snr[toolow]   = 1.0e-5
                    err[toolow]   = 2.0 * np.median( block )

                    # Set up a fit of the order position on the block
                    pguess = np.array( [ np.argmax( block[0,:] ), 0.0, 0.0,
                                    np.median( block[:,np.argmax( block[1000,:] )] ), 0.0, 0.0,
                                    2.0, 0.0, 0.0, 0.0, np.percentile( block, 2 ) ] )

                    params, pcov = optim.curve_fit( Order_Model_2D, ( x, y ), block.ravel(), p0 = pguess, sigma = err.ravel() )

                    model, mean, peak, sigma = Order_Model_2D( ( x, y ), params )
                    resids = ( block - model ) / err

                    residcut      = np.percentile( resids, 99.99 )
                    badcut        = np.where( resids > residcut )
                    block[badcut] = np.median( block )
                    snr[badcut]   = 1.0e-5

                    slowtrace     = mean[:,0].copy()
                    slowpeak      = peak[:,0].copy()
                    slowsigma     = sigma[:,0].copy()
                    slowshape     = np.sum( model, axis = 1 )
                    blockbg       = params[10]
                    blocksigma    = params[5]

                #print "Extracting Ord " + str(ord+1) +" out of " + str(trace.shape[0]) + " for frame " + str(frm+1) + " of " + str(cube.shape[0])

                savesigma = np.zeros( Trace.shape[1] )
                savepeak  = np.zeros( Trace.shape[1] )
                savebg    = np.zeros( Trace.shape[1] )
                savespot  = np.zeros( Trace.shape[1] )

                for pix in range( Trace.shape[1] ):

                    sliceval   = block[pix,:]
                    slicesnr   = snr[pix,:]
                    slicex     = np.arange( len( sliceval ) )
                    slicenoise = np.absolute( sliceval / slicesnr )
                    negsnr     = np.where( slicesnr < 0.0 )[0]
                    if len( negsnr ) >= 1:
                        pdb.set_trace()'''

########## WAVELENGTH CALIBRATION FUNCTIONS AND WHAT NOT ##########

def Gaussian( x, A, mean, sigma, const ):
    '''
    Returns a gaussian function for fitting purposes with scipy.optimize.curve_fit
    '''
    
    gauss = A * np.exp( - ( x - mean ) ** 2.0 / ( 2.0 * sigma ** 2 ) ) + const

    return gauss

def Get_WavSol( Cube, RoughSol, rdir, codedir, plots = True, Orders = 'All' ):
    
    if not os.path.exists( rdir + 'wavcal' ):
        os.mkdir( rdir + 'wavcal' )
    
    if Orders == 'All':
        orderloop = range( Cube.shape[1] )
    else:
        orderloop = Orders

    #orderdif = RoughSol.shape[0] - Cube.shape[1]
    orderdif = 1
        
    FullWavSol  = np.zeros( ( Cube.shape[0], Cube.shape[1], Cube.shape[2] ) )
    FullParams  = np.zeros( ( Cube.shape[0], Cube.shape[1], 5 ) )
    
    THAR            = { 'wav': 0, 'spec': 0, 'logspec': 0, 'lines': 0 }
    THARcalib       = fits.open( codedir + 'thar_photron.fits' )[0]
    header          = THARcalib.header
    THAR['spec']    = THARcalib.data
    THAR['wav']     = np.arange( len(THAR['spec']) ) * header['CDELT1'] + header['CRVAL1']
    THAR['logspec'] = np.log10( THAR['spec'] )
    THAR['lines']   = readcol.readcol( codedir + 'ThAr_list.txt', asRecArray = True ).wav
    
    #for frame in range( Cube.shape[0] ):
    for frame in range(1):
        framepath = rdir + 'wavcal/arcframe_' + str(frame)
        if not os.path.exists( framepath ):
            os.mkdir( framepath )
            
        badorders = []
        for order in orderloop:
            orderpath = framepath + '/order_' + str(order)
            if os.path.exists( orderpath ):
                for f in glob.glob( orderpath + '/*' ): os.remove(f)
            else:
                os.mkdir( orderpath )
            
            arcspec   = Cube[frame,order,:]
            prelimsol = RoughSol[order+orderdif,:]
            
            #arcspec         = arcspec - np.min( arcspec )
            logarcspec      = np.log10( arcspec + 1 )
            logarcspec      = logarcspec - np.min( logarcspec )
            
            wavsol, params, keeps, rejs, flag = Fit_WavSol( prelimsol, arcspec, THAR['lines'], orderpath, plots = plots )
            
            if flag:
                badorders.append(order)
                
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
    
    print badorders
    return FullWavSol, FullParams

def Find_Peaks( wav, spec, peaksnr = 5, pwidth = 10, minsep = 0.5 ):
    
    cutspec = spec.copy()
    cut = np.percentile( spec, 90.0 )
    below = spec >= cut
    cutspec[below] = np.median(spec)
    smoothed = signal.savgol_filter( cutspec, 101, 3)
    spec /= smoothed
    
    # Find peaks using the cwt routine from scipy.signal
    peaks = signal.find_peaks_cwt( spec, np.arange( 2, 4 ), min_snr = peaksnr, noise_perc = 20 )
    
    # Offset from start/end of spectrum by some number of pixels
    peaks = peaks[ (peaks > pwidth) & (peaks < len(spec) - pwidth) ]
    
    pixcent = np.array([])
    wavcent = np.array([])
        
    for peak in peaks:
        
        xi   = wav[peak - pwidth:peak + pwidth]
        yi   = spec[peak - pwidth:peak + pwidth]
        inds = np.arange( len(xi), dtype = float )
        
        pguess   = [ yi[9], np.median( inds ), 0.9, np.median( spec ) ]
        lowerbds = [ 0.1 * pguess[0], pguess[1] - 2.0, 0.3, 0.0  ]
        upperbds = [ np.inf, pguess[1] + 2.0, 1.5, np.inf ]
        print pguess
        print peak
        if xi[0] > 8424.0:
            plt.plot( np.log10(spec), 'k-' )
            plt.show()
            plt.plot( xi, yi, 'k-' )
            plt.axvline( x = np.mean([xi[9],xi[10]]), color = 'r' )
            plt.show()
        try:
            params, pcov = optim.curve_fit( Gaussian, inds, yi, p0 = pguess, bounds = (lowerbds,upperbds) )
            
            pixval  = peak - pwidth + params[1]
            pixcent = np.append( pixcent, pixval )
            
            ceiling = np.ceil( pixval ).astype(int)
            floor   = np.floor( pixval ).astype(int)
            slope   = ( wav[ceiling] - wav[floor] ) / ( ceiling - floor )
            wavval  = wav[floor] + slope * ( pixval - floor )
            wavcent = np.append( wavcent, wavval )
            
        except RuntimeError:
            pixval  = 'nan'
    
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

def Fit_WavSol( wav, spec, THARcat, path, snr = 5, minsep = 0.5, plots = True ):
    
    wavsol = np.zeros( len(spec), dtype = np.float64 )
    
    pixcent, wavcent = Find_Peaks( wav, spec, peaksnr = snr, minsep = minsep )
    
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
    cutoff = 3.0# / 0.67449 # Corrects MAD to become sigma
    
    while dofit:
        wavparams  = np.polyfit( keeps['pix'], keeps['line'], 4 )
        ptsfromfit = np.polyval( wavparams, keeps['pix'] )
        
        wavresids  = ptsfromfit - keeps['line']
        velresids  = wavresids / keeps['line'] * 3e5
        resids     = { 'wav': wavresids, 'vel': velresids }
        
        velcut = np.sum( np.abs(resids['vel']) >= 0.2 )
        torej  = np.abs( resids['wav'] ) >= cutoff * np.median( np.abs( resids['wav'] ) )
        tokeep = np.logical_not( torej )
        numrej = np.sum( torej )
        
        if velcut > 0:
            if numrej > 0:
                if plots:
                    plotname = path + '/resids_round_' + str(ploti) + '.pdf'
                    Plot_WavSol_Resids( resids, keeps['line'], cutoff, plotname, tokeep = tokeep, toreject = torej )
                
                rejs['pix']  = keeps['pix'][torej]
                rejs['wav']  = keeps['wav'][torej]
                rejs['line'] = keeps['line'][torej]
                
                keeps['pix']  = keeps['pix'][tokeep]
                keeps['wav']  = keeps['wav'][tokeep]
                keeps['line'] = keeps['line'][tokeep]
                
                ploti += 1
                
            elif numrej == 0 and cutoff == 3.0:
                cutoff = 2.0
                
            else:
                print 'There is something seriously wrong.\n'
                print 'There are points > 1 km/s, but none are found to be rejected. FIX'
                flag = True
                if plots:
                    plotname = path + '/resids_round_' + str(ploti) + '_flag.pdf'
                    Plot_WavSol_Resids( resids, keeps['line'], cutoff, plotname )
                break

        else:
            if plots:
                plotname = path + '/resids_round_' + str(ploti) + '.pdf'
                Plot_WavSol_Resids( resids, keeps['line'], cutoff, plotname )
            flag = False
            dofit = False
            
    wavsol = np.polyval( wavparams, np.arange( len(spec) ) )

    return wavsol, wavparams, keeps, rejs, flag

def Plot_WavSol_Resids( resids, lines, cutoff, savename, tokeep = None, toreject = None ):
    
    plt.clf()
    fig, (wavax, velax) = plt.subplots( 2, 1, sharex = 'all' )
    wavax.axhline( y = 0.0, color = 'k', ls = ':' )
    velax.axhline( y = 0.0, color = 'k', ls = ':' )
    if toreject == None:
        wavax.plot( lines, resids['wav'], 'ko', mfc = 'none' )
        velax.plot( lines, resids['vel'], 'ko', mfc = 'none' )
        fig.suptitle( 'Lines Used: ' + str(lines.size) + ', Cutoff: ' + str(cutoff * 0.67449) + ' $\sigma$' )
    else:
        wavax.plot( lines[tokeep], resids['wav'][tokeep], 'ko', mfc = 'none' )
        wavax.plot( lines[toreject], resids['wav'][toreject], 'kx' )
        velax.plot( lines[tokeep], resids['vel'][tokeep], 'ko', mfc = 'none' )
        velax.plot( lines[toreject], resids['vel'][toreject], 'kx' )
        fig.suptitle( 'Lines Used: ' + str(lines[tokeep].size) + ', Lines Rej: ' + str(lines[toreject].size) 
                     + ', Cutoff: ' + str(cutoff * 0.67449) +  ' $\sigma$' )
    for x in [ -cutoff, cutoff ]:
        wavax.axhline( y = x * np.median( np.abs( resids['wav'] ) ), color = 'r', ls = '--' )
        velax.axhline( y = x * np.median( np.abs( resids['vel'] ) ), color = 'r', ls = '--' )
    wavax.set_ylabel( 'Resids ($\AA$)' )
    velax.set_ylabel( 'Resids (km/s)' )
    wavax.yaxis.set_label_position("right"); velax.yaxis.set_label_position("right")
    fig.subplots_adjust( hspace = 0 )
    plt.savefig(savename)
    
    return None

def Plot_Wavsol_Windows( wavsol, wavkeep, spec, THAR, path, frame, order ):
    
    numfigs = np.ceil( np.ceil( np.ptp( wavsol ) / 10.0 ) / 6.0 ).astype(int)
    start   = np.min( wavsol )
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
        plt.savefig( path + '/specwindow_' + str(i) + '.pdf' )
    
    return None
