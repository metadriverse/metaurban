#include "const.h"

#ifndef ORCA_ENVIRONMENTOPTIONS_H
#define ORCA_ENVIRONMENTOPTIONS_H

enum MAPFTriggers {
	COMMON_POINT,
	SPEED_BUFFER
};


class environment_options {
	public:
		environment_options() = default;

		environment_options(const environment_options &obj);

		environment_options(int mt, bool bt, bool as, bool cc, float hw, float ts, float del, MAPFTriggers tr, int mn);

		int metrictype;     //Can be chosen Euclidean, Manhattan, Chebyshev and Diagonal distance
		bool breakingties;   //Option that defines the priority in OPEN list for nodes with equal f-values
		bool allowsqueeze;   //Option that allows to move through "bottleneck"
		bool cutcorners;     //Option that allows to make diagonal moves, when one adjacent cell is untraversable
		float hweight;        //Option that defines the weight of the heuristic function
		float timestep;       //Option that defines time per one step of simulation
		float delta;          //Option that defines max distance to goal position, which is considered as reaching the goal position
		MAPFTriggers trigger;   //Option that defines type of condition for transition to mapf mode
		int MAPFNum;            //Option that defines the number of neighbours, which are taken into account in mapf trigger

};


#endif //ORCA_ENVIRONMENTOPTIONS_H
