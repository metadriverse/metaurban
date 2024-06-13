#include <tuple>
#include <set>
#include <unordered_set>
#include <iostream>
#include <vector>

#include "../const.h"

#ifndef ORCA_CONSTRAINTS_H
#define ORCA_CONSTRAINTS_H


struct Constraint {
	int i, j; //grid cell coordinates
	int time;
	int dur;
	int agentId;
	int prev_i, prev_j;
	bool goalNode;
	bool positive;

	Constraint(int x = 0, int y = 0, int Time = 0, int AgentId = 0, int PrevI = -1, int PrevJ = -1,
			   bool GoalNode = false) {
		i = x;
		j = y;
		time = Time;
		agentId = AgentId;
		prev_i = PrevI;
		prev_j = PrevJ;
		goalNode = GoalNode;
		positive = false;
		dur = 1;
	}

	bool operator==(const Constraint &other) const {
		return i == other.i && j == other.j && time == other.time &&
			   prev_i == other.prev_i && prev_j == other.prev_j && goalNode == other.goalNode;
	}

	bool operator!=(const Constraint &other) const {
		return !(*this == other);
	}

	bool operator<(const Constraint &other) const {
		return std::tuple<int, int, int, int, int, bool>(i, j, prev_i, prev_j, time, goalNode) <
			   std::tuple<int, int, int, int, int, bool>(other.i, other.j, other.prev_i, other.prev_j, other.time,
														 other.goalNode);
	}
};

class ConstraintsSet {
	public:
		void addNodeConstraint(int i, int j, int time, int agentId);

		void addGoalNodeConstraint(int i, int j, int time, int agentId);

		void addEdgeConstraint(int i, int j, int time, int agentId, int prevI, int prevJ);

		void addPositiveConstraint(int i, int j, int time, int agentId, int prevI = -1, int prevJ = -1);

		void addConstraint(Constraint &constraint);

		template<typename Iter>
		void addAgentPath(Iter start, Iter end, int agentId);

		void removeNodeConstraint(int i, int j, int time, int agentId);

		void removeGoalNodeConstraint(int i, int j, int time, int agentId);

		void removeEdgeConstraint(int i, int j, int time, int agentId, int prevI, int prevJ);

		template<typename Iter>
		void removeAgentPath(Iter start, Iter end, int agentId);

		ConstraintsSet getAgentConstraints(int agentId) const;

		std::vector<Constraint> getPositiveConstraints() const;

		int getFirstConstraintTime(int i, int j, int startTime, int agentId) const;

		std::vector<std::pair<int, int>> getSafeIntervals(int i, int j, int agentId, int startTime, int endTime) const;

		bool hasNodeConstraint(int i, int j, int time, int agentId) const;

		bool hasFutureConstraint(int i, int j, int time, int agentId) const;

		bool hasEdgeConstraint(int i, int j, int time, int agentId, int prevI, int prevJ) const;

//private:
		std::set<Constraint> nodeConstraints;
		std::set<Constraint> edgeConstraints;
		std::set<Constraint> goalNodeConstraints;
		std::vector<Constraint> positiveConstraints;
};

template<typename Iter>
void ConstraintsSet::addAgentPath(Iter start, Iter end, int agentId) {
	int time = 0;
	for (auto it = start; it != end; ++it) {
		if (std::next(it) == end) {
			addGoalNodeConstraint(it->i, it->j, time, agentId);
		}
		else {
			addNodeConstraint(it->i, it->j, time, agentId);
		}
		if (it != start) {
			addEdgeConstraint(std::prev(it)->i, std::prev(it)->j, time, agentId, it->i, it->j);
		}
		++time;
	}
}

template<typename Iter>
void ConstraintsSet::removeAgentPath(Iter start, Iter end, int agentId) {
	int time = 0;
	for (auto it = start; it != end; ++it) {
		if (std::next(it) == end) {
			removeGoalNodeConstraint(it->i, it->j, time, agentId);
		}
		else {
			removeNodeConstraint(it->i, it->j, time, agentId);
		}
		if (it != start) {
			removeEdgeConstraint(std::prev(it)->i, std::prev(it)->j, time, agentId, it->i, it->j);
		}
		++time;
	}
}


#endif //ORCA_CONSTRAINTS_H
