# KCWI_Target_Selection_Tests
Some scripts to help identify CRD targets and their specific KCWI instrument observing settings

I'd eventually like to make this a direct pipeline where you input your CRD target in the target_config.txt file and hit go, and it then runs through all 5 scripts to tell you how well it could recover the blue arm population, red arm, blue arm given the red arm kinematics, how dependent it is on the fixed light fraction of each disk, and finally what the geometry of a first-pass Voronoi binning scheme would look like for that specific galaxy. 

Note that for the Voronoi binning test script, it needs the specific LOGCUBE file from MaNGA to read the surface brightness to estimate S/N given KCWI instrument settings.
