How to run:
1) Edit the macro in search.cu to define the value of parameter K, where you should write the number
of classes of the wanted dataset.
2) Make all
3) usage: ./search <dataset> <d> <iterations>
where <dataset> is the wanted dataset, <d> is the dimensionality of its feature space, and <iterations>
is the number of serial iterations.

The following illustrates all possible choices:
	*Set1 	   2 	8
	*Set22	   2	8
	*Pavia           35	8
	*PaviaU        36             8
	*Salinas       53	8
	*SalinasA    63             8

Note that 8 is the default number of iterations we are implementing in the code. Changing it at will 
is easy. 

**IMPORTANT FOR CHANGING THE NUMBER OF ITERATIONS**
1) To change the number of serial iterations, in addition to entering the required number in the command 
line, please make sure that the entered number is a power of 2.
2) The macro (serial_iteration) in search.cu should be changed to match the entered value.

**To check the correctness of the launch**
*If the correct d is entered according to the above table in the command line, then the correct number 
of samples will be written in the command window. The numbers of samples per dataset are:
	*Set1 	   110,000
	*Set2	   160,000
	*Pavia           148,152
    	*PaviaU        42,776
	*Salinas       54,129
	*SalinasA    5348
	  
*Any mistake in the command line will result in a (core dumped) error. If everything is correct, the 
program should run successfully.

*All results of the different iterations, along with the error logs can be found in ./log
*Notice that the output groups assume 1-indexing, and a wrapped array (non-flattened) i.e. MATLAB
style indexing. Therefore, the labeling of each group can be obtained directly in MATLAB using 
	real_labels = Y(g);
where g = [ output group array from CUDA];
*The MATLAB files (.mat) of all used datasest with the used shuffling exist as well.
