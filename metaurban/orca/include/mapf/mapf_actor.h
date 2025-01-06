#include "../geom.h"

#ifndef ORCA_PARACTOR_H
#define ORCA_PARACTOR_H

class MAPFActor {
	public:

		MAPFActor(int _start_i = 0, int _start_j = 0, int _goal_i = 0, int _goal_j = 0, int _id = 0) {
			start_i = _start_i;
			start_j = _start_j;
			goal_i = _goal_i;
			goal_j = _goal_j;
			cur_i = _start_i;
			cur_j = _start_j;
			id = _id;
			subgraph = -1;
		}

		int getStart_i() const;

		int getStart_j() const;

		int getGoal_i() const;

		int getGoal_j() const;

		int getCur_i() const;

		int getCur_j() const;

		int getId() const;

		int getSubgraph() const;

		Node getStartPosition() const;

		Node getGoalPosition() const;

		Node getCurPosition() const;

		void setCurPosition(int i, int j);

		void setSubgraph(int Subgraph);

	private:
		int id;
		int start_i, start_j;
		int goal_i, goal_j;
		int cur_i, cur_j;
		int subgraph;
};


#endif //ORCA_PARACTOR_H
