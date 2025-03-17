#include "scheduler.h"

/*************************************************************************

scheduler.cpp

This file is a wrapper for all the other functions that the scheduler
needs to be able to successfully handle the reschedule requests.

The file was split into 3 distinct parts once the CUDA additions
made it difficult to quickly move through the file.

Typically splitting into separate classes would be done here, but
these functions really are part of this class and have no use anywhere
else, so it feels equally weird to split them into separate classes.

This may change into just doing this in the makefile later, but for now
I can leave a message here and it feels more natural to see which files
are being included in the main scheduler object directly in the source.

**************************************************************************/
#include "knapsack_functions.cpp"
#include "coordination_functions.cpp"
#include "graph_functions.cpp"