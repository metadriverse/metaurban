#include <vector>
#include <unordered_map>
#include <list>
#include "../geom.h"
#include "constraints.h"
#include "mdd.h"
#include "conflict_set.h"

#ifndef ORCASTAR_CBSNODE_H
#define ORCASTAR_CBSNODE_H


struct CBSNode {
	static int curId;

	Constraint constraint, positiveConstraint;
	std::unordered_map<int, std::list<Node>> paths;
	std::unordered_map<int, MDD> mdds;
	std::unordered_map<int, double> lb;
	double sumLb;
	int hc;
	CBSNode *parent;
	ConflictSet conflictSet;
	int cost;
	int id;
	int H, G;
	bool hasPositiveConstraint;
	bool pathFound;

	CBSNode(CBSNode *p = nullptr, int Cost = 0) {
		parent = p;
		cost = Cost;
		id = curId++;
		H = 0;
		hasPositiveConstraint = false;
		pathFound = true;
		hc = 0;
		sumLb = 0;
	}

	CBSNode(bool PathFound) {
		pathFound = PathFound;
	}


	bool operator<(const CBSNode &other) const {
		return G < other.G || (G == other.G && id < other.id);
	}
};

#endif //ORCASTAR_CBSNODE_H
