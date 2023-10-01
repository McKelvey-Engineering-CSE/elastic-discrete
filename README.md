# elastic-discrete

Author: James Orr
Date: 5 September, 2018

This is a runtime system to run parallel elastic tasks with discrete candidate values of period T OR work C. 
Each task Tau has a constant span L, variable T or C, Elasticity coefficient E, and a finite set of discrete values of C or T

See example.rtps for taskset representation.

In order to get this running on Cybermech run the following 2 commands each session (also in init.sh): 
export PATH=/home/james/bin:$PATH &&  export LD_LIBRARY_PATH=/home/james/lib64:$LD_LIBRARY_PATH && export GOMP_SPINCOUNT=0
