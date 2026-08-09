# KCWI_Target_Selection_Tests
Some scripts to help identify CRD targets and their specific KCWI instrument observing settings

I'd eventually like to make this a direct pipeline where you input your CRD target in the target_config.txt file and hit go, and it then runs through all 5 scripts to tell you how well it could recover the blue arm population, red arm, blue arm given the red arm kinematics, how dependent it is on the fixed light fraction of each disk, and finally what the geometry of a first-pass Voronoi binning scheme would look like for that specific galaxy. 

Note that for the Voronoi binning test script, it needs the specific LOGCUBE file from MaNGA to read the surface brightness to estimate S/N given KCWI instrument settings.

Further notes on each script:
# KCWI_CRD_injection_recovery_refined_configfile
This is the core/base two-component kinematic simulation, using KCWI BL. It asks: Given the target's expected V_A, V_B, sigma_A, sigma_B, light ratio, selected slicer, S/N target, can BL distinguish and recover the two counter-rotating stellar LOSVDs? It constructs synthetic XSL spectra containing two stellar components with known velocities, dispersions, and light contributions; degrades them to the BL resolution appropriate for Small/Medium/Large; adds Gaussian noise; and then performs the brute-force V_A x V_B pPXF search. 

For every noisy spectrum it also fits a one-component model, so it can calculate Delta chi^2 = chi^2_comp1 - chi^2_comp2. It generates true one-component spectra as a null experiment and uses the 95th percentile of that null Delta chi^2 distribution as the detection threshold. It then evaluates three increasingly demanding questions: (1) Do I detect two LOSVDs? (2) Do I recover Delta V accurately? (3) Do I recover V_A and V_B individually?
