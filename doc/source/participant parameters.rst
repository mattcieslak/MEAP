.. _participant-parameters: 

Participant Parameters
========================

.. toctree::
   :hidden:

These are variables you can include. 
	1. **age** 
	2. **gender** M= male, F= female
	3. weight = lbs
	4. height_ft = Subject's height in feet
	5. height_in = Subject's height in inches
	6. electrode_distance_front = Impedance electrode distance (front)
	7. electrode_distance_back = Impedance electrode distance (back)
	8. electrode_distance_right = Impedance electrode distance (right)
	9. electrode_distance_left =Impedance electrode distance (left)
	10. resp_max = Respiration circumference max (cm)
	11. resp_min = Respiration circumference min (cm)
	12. in_mri = True or False
	13. control_base_impedance = If in MRI, stores the z0 value from outside the MRI since this 
		value is impacted by scanning. 
		
Equations: 
			
	+-----------------------------+-------------------------------+------------------------+ 
    |         Measure             |  Definition                   | Calculation Method     |     
    |                             |                               |                        |           
    +=============================+===============================+========================+ 
    | Total Peripheral Resistance | The total resistance of the   |                        | 
    | (TPR)                       | body's peripheral vasculature |   (MAP/CO) x 80        | 
    |                             |                               |                        | 
    +-----------------------------+-------------------------------+------------------------+ 
    | Cardiac Output  (CO)        | The amount of blood pumped    |    (SV x HR)/1000      | 
    |                             | by the heart per minute       |                        | 
    |                             |                               |                        | 
    +-----------------------------+-------------------------------+------------------------+ 
    | Stroke Volume (SV)          | Volume of blood pumped        | Area under B to X on   | 
    |                             | each beat                     | dz/dt waveform         | 
    |                             |                               |                        | 
    +-----------------------------+-------------------------------+------------------------+ 
    | Pre-Ejection Period (PEP)   | aka Ventricular Contractility | Time between R and B   | 
    |                             | (VC) the force with which the |                        | 
    |                             | left ventrical contracts      |                        | 
    +-----------------------------+-------------------------------+------------------------+ 
    | Mean Arterial Pressure      | Average arterial pressure     | 2/3(DBP) + 1/3(SBP)    | 
    | (MAP)                       | during a cardiac cycle        | DBP=diastolic          | 
    |                             |                               | SBP=systolic           | 
    +-----------------------------+-------------------------------+------------------------+ 
    | Heart Rate  (HR)            | Beats per minute              | # of peaks in the      | 
    |                             |                               | waveform per minute    | 
    |                             |                               |                        | 
    +-----------------------------+-------------------------------+------------------------+ 