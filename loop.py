"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!

Modified by Shreya and Phoebe

Gaussian: Yes
Animation: No
Writes to: focus.txt (could be changed)
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import cv2
import scipy.optimize
import time

#initialize data lists
testdata = []
centerx = []
centery = []
sigx = []
sigy = []
x = []

# functions needed for fitting and stuff
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple 
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def iter_Gauss(data):

    shape = data.shape
    print('shape', shape)
    # Create x and y indices
    x=range(shape[1])
    y=range(shape[0])
    #print(x,y)
    x,y = np.meshgrid(x,y)

    # plot twoD_Gaussian data generated above
    #plt.figure()
    #plt.imshow(data)
    #plt.colorbar()

    # add some noise to the data and try to fit the data generated beforehand
    initial_guess = (250,shape[1]/2,shape[0]/2,shape[1]/2,shape[0]/2,0,1) #amp, centerx, centery, sigx,sigy, theta, offset

    popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian, (x,y), data.reshape(shape[0]*shape[1]), p0 = initial_guess)
    #print('{}:'.format(i), popt)
    #centerx.append(popt[1]) #currently not in use
    #centery.append(popt[2]) #currently not in use
    #sigx.append(popt[3])
    #sigy.append(popt[4])
    return (popt[0],popt[3], popt[4],popt[5],popt[6]) #optional

#open the video
cap = cv2.VideoCapture('movies/vlc-100106-115345.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/1000) #get number of frames
print('length', length)

# First set up the figure, the axis, and the plot element we want to animate
#fig = plt.figure()
#ax = plt.axes(xlim = (0,length), ylim=(0, 10))
#line, = ax.plot([], [], lw=2, marker = '.')

#open the video
cap = cv2.VideoCapture('movies/vlc-100106-115345.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# initialization function: plot the background of each frame
#def init():
#    line.set_data([], [])
#    return line,

# animation function.  This is called sequentially
# this is where the physics goes, hopefully
startTime = time.time()
i = 0
while True:
    #grab the next frame
    ret, frame = cap.read()
    if i % 5 == 0:
       # implement the stuff from the while loop
       data = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:200] #indexing crops off the bottom

       # gaussian blur reduces noise, hopefully gets rid of hot pixels
       # only use 'blurred' for blob detection
       blurred = cv2.GaussianBlur(data, (3,3), 0)
       # reverse because blob detection is optimized to detect dark blobs
       reverseImg = 255-blurred

       # Set up the SimpleBlobdetector with default parameters.
       params = cv2.SimpleBlobDetector_Params()

       # Change thresholds
       params.minThreshold = 0;
       params.maxThreshold = 256;

       # Filter by Area.
       params.filterByArea = True
       params.minArea = 30

       # Filter by Circularity
       params.filterByCircularity = True
       params.minCircularity = 0.1

       # Filter by Convexity
       params.filterByConvexity = True
       params.minConvexity = 0.5

       # Filter by Inertia
       params.filterByInertia = True
       params.minInertiaRatio = 0.5

       #make blob detection object
       detector = cv2.SimpleBlobDetector_create(params)

       # Detect blobs.
       keypoints = detector.detect(reverseImg) #keypoints is a length 1 list (hopefully)

       # extract blob coordinates
       blobx, bloby = keypoints[0].pt
       print('blobx:', blobx, 'bloby:', bloby)
       # set cropping limits

       rad = int(keypoints[0].size/2)
       #padding = int(0.25*diam)

       #set cropping limits
       xmin = int(blobx) - 2*rad 
       xmax = int(blobx) + 2*rad 
       ymin = int(bloby) - 2*rad
       ymax = int(bloby) + 2*rad

   
       # call the fitting function defined above
       amp, sigx, sigy,theta,offset = iter_Gauss(data[ymin:ymax, xmin:xmax])

       #calculate focus from sigx and sigy
       foc = np.sqrt(sigx**2 + sigy**2)
       testdata.append(foc)
       print('focus', foc)

       timeSinceStart = time.time() - startTime

       # append line to datafile
       with open('focus.txt', 'a') as outfile:
       	    outfile.write('{},{},{},{},{},{},{},{},{},{}\n'.format(i,timeSinceStart,foc,blobx,bloby,sigx,sigy,amp,theta,offset))

	# x = np.arange(0, len(testdata), 1)
       
       x.append(timeSinceStart)
    i += 1
       #print(timeSinceStart)
       #y = testdata
       #line.set_data(x, y)
       #ax.set_xlim(0, x[-1])
       #return line,

# save start time
#startTime = time.time()

# call the animator.  blit=True means only re-draw the parts that have changed.
#anim = animation.FuncAnimation(fig, animate, init_func=init,
#                               frames=(length - 1), interval=25, blit=False, save_count = length)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

#plt.show()

