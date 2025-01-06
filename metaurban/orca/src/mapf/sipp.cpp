#include "mapf/sipp.h"


template<typename NodeType>
void SIPP<NodeType>::setEndTime(NodeType &node, int start_i, int start_j, int startTime, int agentId,
								const ConstraintsSet &constraints) {
	node.endTime = constraints.getFirstConstraintTime(start_i, start_j, startTime, agentId);
	if (node.endTime < CN_INFINITY) {
		--node.endTime;
	}
}

template<typename NodeType>
void SIPP<NodeType>::updateEndTimeBySoftConflicts(NodeType &node, const ConflictAvoidanceTable &CAT) {
	int newEndTime = CAT.getFirstSoftConflict(node, node.startTime, node.endTime);
	if (newEndTime != -1) {
		node.endTime = newEndTime - 1;
	}
}

template<typename NodeType>
void SIPP<NodeType>::createSuccessorsFromNode(const NodeType &cur, NodeType &neigh, std::list<NodeType> &successors,
											  int agentId, const ConstraintsSet &constraints,
											  const ConflictAvoidanceTable &CAT, bool isGoal) {
	std::vector<std::pair<int, int>> safeIntervals = constraints.getSafeIntervals(
			neigh.i, neigh.j, agentId, cur.g + 1,
			cur.endTime + (cur.endTime != CN_INFINITY));

	for (auto interval: safeIntervals) {
		addOptimalNode(cur, neigh, interval, agentId, constraints, successors);
		setOptimal(neigh, false);
		std::vector<std::pair<int, int>> softConflictIntervals;
		splitBySoftConflicts(softConflictIntervals, neigh, cur, interval, CAT);
		if (checkSuboptimal(cur)) {
			for (int i = 0; i < softConflictIntervals.size(); ++i) {
				if (!withZeroConflicts() || (softConflictIntervals[i].second == 0 &&
											 (!isGoal || i == softConflictIntervals.size() - 1))) {
					neigh.startTime = softConflictIntervals[i].first;
					neigh.endTime = (i == softConflictIntervals.size() - 1) ? interval.second :
									softConflictIntervals[i + 1].first - 1;
					neigh.conflictsCount = softConflictIntervals[i].second;
					setNeighG(cur, neigh, agentId, constraints);
					this->setHC(neigh, cur, CAT, isGoal);
					if (neigh.g <= cur.endTime + 1 && neigh.g <= neigh.endTime) {
						neigh.F = neigh.g + neigh.H;
						successors.push_back(neigh);
					}
				}
			}
		}
	}
}

template<typename NodeType>
void SIPP<NodeType>::setNeighG(const NodeType &cur, NodeType &neigh,
							   int agentId, const ConstraintsSet &constraints) {

	for (neigh.g = std::max((double) neigh.startTime, (double) cur.g + 1); neigh.g <= neigh.endTime; ++neigh.g) {
		if (!constraints.hasEdgeConstraint(neigh.i, neigh.j, neigh.g, agentId, cur.i, cur.j)) {
			break;
		}
	}
}

template<typename NodeType>
void SIPP<NodeType>::splitBySoftConflicts(std::vector<std::pair<int, int>> &softConflictIntervals,
										  const NodeType &node, const NodeType &prevNode, std::pair<int, int> interval,
										  const ConflictAvoidanceTable &CAT) {
	softConflictIntervals.push_back(std::make_pair(interval.first, 0));
}

template<typename NodeType>
bool SIPP<NodeType>::checkGoal(const NodeType &cur, int goalTime, int agentId, const ConstraintsSet &constraints) {
	return goalTime == -1 || cur.g <= goalTime;
}

template
class SIPP<SIPPNode>;

template
class SIPP<SCIPPNode>;
