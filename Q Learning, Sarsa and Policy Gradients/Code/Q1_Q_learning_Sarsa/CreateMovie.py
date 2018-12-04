def CreateMovie(moviefilename,fps=5):
	import os, sys
	os.system("rm "+str(moviefilename)+".mp4")
	os.system("ffmpeg -r "+str(fps) +
	          " -b 1800 -i _tmp%05d.png "+str(moviefilename)+".mp4")
	os.system("rm _tmp*.png")
