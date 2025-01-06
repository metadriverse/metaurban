#include "mapf/mapf_actor.h"

int MAPFActor::getStart_i() const {
	return start_i;
}

int MAPFActor::getStart_j() const {
	return start_j;
}

int MAPFActor::getGoal_i() const {
	return goal_i;
}

int MAPFActor::getGoal_j() const {
	return goal_j;
}

int MAPFActor::getCur_i() const {
	return cur_i;
}

int MAPFActor::getCur_j() const {
	return cur_j;
}


int MAPFActor::getId() const {
	return id;
}

int MAPFActor::getSubgraph() const {
	return subgraph;
}

Node MAPFActor::getStartPosition() const {
	return Node(start_i, start_j);
}

Node MAPFActor::getGoalPosition() const {
	return Node(goal_i, goal_j);
}

Node MAPFActor::getCurPosition() const {
	return Node(cur_i, cur_j);
}

void MAPFActor::setCurPosition(int i, int j) {
	cur_i = i;
	cur_j = j;
}

void MAPFActor::setSubgraph(int Subgraph) {
	subgraph = Subgraph;
}

