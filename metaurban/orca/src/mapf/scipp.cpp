#include "mapf/scipp.h"


template<typename NodeType>
SCIPP<NodeType>::~SCIPP() {}

template<typename NodeType>
void SCIPP<NodeType>::splitBySoftConflicts(std::vector<std::pair<int, int>> &softConflictIntervals,
										   const NodeType &node, const NodeType &prevNode, std::pair<int, int> interval,
										   const ConflictAvoidanceTable &CAT) {
	CAT.getSoftConflictIntervals(softConflictIntervals, node, prevNode, interval.first, interval.second, false);
}

template<typename NodeType>
void SCIPP<NodeType>::addStartNode(NodeType &node, const SubMap &map, const ConflictAvoidanceTable &CAT) {
	this->updateEndTimeBySoftConflicts(node, CAT);
	this->open.insert(map, node, ISearch<NodeType>::withTime);
}

template<typename NodeType>
void SCIPP<NodeType>::setHC(NodeType &neigh, const NodeType &cur,
							const ConflictAvoidanceTable &CAT, bool isGoal) {
	neigh.hc = cur.hc + cur.conflictsCount * (neigh.g - cur.g - 1) + neigh.conflictsCount;
	if (isGoal) {
		this->addFutureConflicts(neigh, CAT);
	}
}

template
class SCIPP<SCIPPNode>;

template
class SIPP<SIPPNode>;
