# CRD_KCWI_TargetSelectionTests
Some scripts to help identify CRD targets and their specific KCWI instrument observing settings

I'd eventually like to make this a direct pipeline where you input your CRD target in the target_config.txt file and hit go, and it then runs through all 5 scripts to tell you how well it could recover the blue arm population, red arm, blue arm given the red arm kinematics, how dependent it is on the fixed light fraction of each disk, and finally what the geometry of a first-pass Voronoi binning scheme would look like for that specific galaxy. 

Note that for the Voronoi binning test script, it needs the specific LOGCUBE file from MaNGA to read the surface brightness to estimate S/N given KCWI instrument settings.

--------------------------------
Further notes on each script:
# KCWI_CRD_injection_recovery_refined_configfile.py
This is the core/base two-component kinematic simulation, using KCWI BL. It asks: Given the target's expected V_A, V_B, σ_A, σ_B, light ratio, selected slicer, S/N target, can BL distinguish and recover the two counter-rotating stellar LOSVDs? 

It constructs synthetic XSL spectra containing two stellar components with known velocities, dispersions, and light contributions; degrades them to the BL resolution appropriate for Small/Medium/Large; adds Gaussian noise; and then performs the brute-force V_A x V_B pPXF search. 

For every noisy spectrum it also fits a one-component model, so it can calculate Δ χ^2 = χ^2_comp1 - χ^2_comp2. It generates true one-component spectra as a null experiment and uses the 95th percentile of that null Δ χ^2 distribution as the detection threshold. It then evaluates three increasingly demanding questions: (1) Do I detect two LOSVDs? (2) Do I recover Δ V accurately? (3) Do I recover V_A and V_B individually?

# KCWI_CRD_Fixed_lightfraction_test_configfile.py
This is a controlled diagnostic version of the prior script. It asks: Was our difficulty recovering the individual absolute velocities caused by insufficient spectral information, or by the degeneracy between light fraction and velocity midpoint? 

In the normal two-component experiment, pPXF can change the relative amount of light in Disk A and Disk B. We found that ΔV could be recovered accurately while both velocities shifted together by ∼15−20 km/s. The hypothesis was that pPXF could partially trade f_A against (V_A + V_B)/2.

This script removes that freedom by fixing the component light fraction while still allowing V_A, V_B, σ_A, σ_B, and the SSP mixturew ithin each disk to vary. It otherwise deliberately keeps the experiment close to the main BL simulation: same wavelength interval, same velocity grid, same null-calibration idea, same S/N values, and matched-control templates.

# KCRM_RM2_RH3_injection_recovery_configfile.py
This is one of the most important kinematic feasibility simulations. It asks: At the selected slicer resolution, what S/N does RH3 require to recover the two disks' V and σ? 

It reuses the machinery in Script 1 but moves the experiment into the KCRM red arm, especially the Ca II triplet region. The synthetic spectra remain in the galaxy rest frame, and the script adjusts the instrumental resolution and spectral sampling for the selected slicer and redshift. 

It performs both the free_fraction and fixed_50_50 experiments for f_A, allowing us to distinguish the intrinsic information content from the f_A-velocity degeneracy. Crucially, it adds explicit velocity-dispersion recovery diagnostics. So unlike the earlier BL experiment, we're not only asking whether V_A and V_B are recovered, but also whether we can make meaningful σ_A(x,y) and σ_B(x,y) maps.

# KCWI_BL_population_recovery_configfile_FIXED.py
This answers a very different question from the prior scripts. It asks: At a given BL S/N, how well can we separate the two disks' stellar populations while fixing V_A, V_B, σ_A, and σ_B from the RH3 kinematics? 

Specifically, it tries to recover the blue light fraction, age of each disk, metallicity of each disk, and whether one component is younger or more metal rich.

It also tests several different population contrasts (for example strongly different ages versus more subtle age/metallicity differences) because population recoverability depends on how spectrally distinct the two actual disks are. Its main outputs include individual Monte Carlo trials, recovery summaries, and diagnostic plots.

# KCWI_Voronoi_binning_test_BL_RH3_configfile.py
Despite the historical filename, this is now our PowerBin spatial/exposure-time planning simulation. It doesn't perform any pPXF fits, but rather asks: Given the actual surface-brightness distribution of this galaxy, the selected slicer, an exposure time, and our required S/N, what spatial sampling should we expect? 

It reads the target's MaNGA data and uses the MaNGA flux distribution as a surface-brightness model. It then places the selected KCWI field at the target's kinematic PA, samples the appropriate Small/Medium/Large geometry, predicts S/N, and adaptively bins the simulated field.

This is the one script that also needs target-specific spatial information from the shared config: PLATEIFU, PA_KIN_DEG, and the approximate radial location of the 2σ along PA_kin from the galactic center. It currently bins to SN_BL = SN_RED = 30. It then runs the simulation over varying exposure times of 2-6hrs and examines things like the number and size of spatial bins, radial coverage, and sampling around the approximate 2σ regions.

# KCRM_RM2_RH3_injection_recovery_fixed_fraction_sweep_configfile.py
This script is to handle an interesting case. It is identical to KCRM_RM2_RH3_injection_recovery_configfile.py, however instead of the fixed-fraction simply being f_A=0.5, it instead varies over f_A = 0.5, 0.6, 0.7, 0.8, and 0.9. I have found that some identified CRDs seem to have only a single visible stellar disk in the MaNGA IFU MAPS. This could be because the MaNGA IFU aperture was too zoomed-in therefore cutting off a fair amount of the galaxy, or it could be the case that the light fraction is just very much unequal and that the weaker stellar disk is much fainter. If the case is the latter, this code tests whether, given the σ_base, σ_peak, and calculated ΔV values from the MaNGA stellar σ MAPS, the LOSVDs are able to be decomposed with the RH3 slicer at varying S/N.  






