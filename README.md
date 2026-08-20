This Repository is built to help future researchers investigate the influence of certain parameters on the DNA circuts system. This repo contains a few folders each with seperate purposes and systems. This was made by Swetan Sinha under the guidance of Dr. Lindeman and Professor Schulman at the ChemBE lab at the Johns Hopkins University. This repo is meant to model the computation conducted in Hydrogels with Tethered Transcription Circuit Elements for Chemical Communication and Collective Computation by Charlie Chen, Professor Schulman and others (ACS 2025). This python repo is a cheaper (but slower) alternative to COMSOL. 


This is primarly meant to simulate the reaction-diffusion process involved in chemical signaling. The order the folders are in below is meant to represent the order they were made

The folders include: 

1. diffusion_exercise: 
      This folder is there in order for someone to become comfortable with programming with FiPy. The exercise involves a numerical simulation which keeps a constant heat source at the center of the large bath as it diffuses. It is then comapred with the analytical solution. There is also an animated option in case one wants to see the process happen.
      
2. DWK_Building_Concentration_Feilds:
      This is meant to be an excercise for building chemical concentrations. It is based on Dr. Kim's "Building Chemical Concentration Feilds (Matter 2025). It first simulates a system in a large bath which has one tethered genelet node in the center. This node produces some chemical sender molecule (in the form of RNA) which diffuses outwards. The equations for rates are in the paper. This then conducts a numerical simulation and compares it to the devised analytical solution. 

3. Simple_Model: 
      The simple model folder is meant to represent the simplest case of this system: a well mixed bath. This model uses the same equation as Chen 25' Section 2.1 except for the primary change in the rate of change of S2. Rather than having a diffusion term + k_p[I1O2], the code is replaced with + Phi_in. This makes the system a well mixed bath with a constant flux of S2. The equations are in their own python file inside of the folder. Of course, please refernce the folder "Final" and not Tests. The "Graphs_of_simple_model" and "well_mixed_results" have results and plots from these simulations

4. Sender_Receiver: 
      This is the system that Chen 25' uses in his COMSOL expirements. However, this covers a more basic case where there are only two nodes communicating with each other. Please refer to the slideshow in order to understand chronologically what was made. Inside this folder are several other folders. Please ignore "failed_things" and only preced to "My_sender_receiver" (CHANGE NAME). The file "preset_parameters.py" has some parameters that certain systems use. 
   
      Inside of My_sender_receiver are six main folders


      a. COMSOL_results: These are files from COMSOL that were used to compare against python results. 
   
      b. Comparision: These are files and plots that compare COMSOL to python results. There are timeseries csvs, python data csvs, plots and scripts that are meant to compare COMSOL and python. "compareCOMSOL_and_python_One_simluation.py" is meant to compare the timeseries of COMSOL and python for one simulation of given parameters. The "COMSOL_vs_Python_parameter_sweep.py" is meant to compare the parameter sweep results for both COMSOL and python. The failed version folder is the previous parameter sweep study I was using which had some errors and was not as accurate as the final version.

      c. Mesh: This contains images of the meshes I have designed and the scripts to make them. New_Simple_mesh.py is the final mesh decided on. This is a triangular mesh designed with Gmsh that radially becomes more coarse. It is very accurate. You can see this mesh imported in other files. Before, I used Grid2D to create a mesh, which was less accurate and slower. 

      d. Functions_and_system: This has the functions, equations and intilizations for the system. This also contains the file "TG_Rmesh_tanh.py" which is the FINAL singular simulation run for this system. This is the most important file. This imports mesh made in New_Simple_mesh.py and uses the functions from Functions.py. You can adjust the parameters desired and watch the print statements tell you the concetrations of the speceies as time goes on. You will also get a final chemical dynamics plot at the end and a print statement in your terminal which tells you some statstics. 

      e. Paramter_Sweep: (FIX SPELLING ERROR) This contains scripts and results of the parameter sweep that I intended to run. There is a failed_tests folder which contains previous parameter sweeps that use teh Grid2D mesh and parameter sweeps that failed becuase of incorrect logic regarding the importing of Gmsh and incorrect solver. The file Parameter_sweep_unified.py is the final parameter sweep file. It contains a section to input any given parameter you want and the values you want to run without having to change it throughout the entire script. Just change it in one place. Mesh_conformal contains the meshes used. There are then several folders which have the results of different parameter sweeps. These include, Threshold variance, center-center distance variance and different rates being varied. 

      f. Convergence_Studies: This contains the convergence studies I tried running. The Convergence_Study_Claudes_Triangular_mesh_FIXEDV4.py is the most recent attempt at running a convergence study, but it is not completed yet. 

I had used Claude Code for debugging, drawings + figures and mesh generation. 


Thank you to everyone who helped me throughout this project. Thank you to Professor Schulman who gave me a space in the office and helped me conduct this research. A big thank you to Dr. Lindeman who met with me almost everyday to answer any questions I had. Additionally, thank you to the office and staff at JHU ChemBE department for helping me in this summer experince. 
