#include "focal_search.h"
#include "sipp.h"
#include "scipp_node.h"

#ifndef ORCASTAR_SCIPP_H
#define ORCASTAR_SCIPP_H


template<typename NodeType = SCIPPNode>
class SCIPP : public SIPP<NodeType>, FocalSearch<NodeType> {
	public:
		SCIPP(double FocalW = 1.0, double HW = 1.0, bool BT = true) :
				Astar<NodeType>(true, HW, BT), SIPP<NodeType>(HW, BT), FocalSearch<NodeType>(true, FocalW, HW, BT) {}

		virtual ~SCIPP();

	protected:
		void splitBySoftConflicts(std::vector<std::pair<int, int>> &softConflictIntervals,
								  const NodeType &node, const NodeType &prevNode, std::pair<int, int> interval,
								  const ConflictAvoidanceTable &CAT) override;

		bool canStay() override { return true; }

		void addStartNode(NodeType &node, const SubMap &map, const ConflictAvoidanceTable &CAT) override;

		void setHC(NodeType &neigh, const NodeType &cur,
				   const ConflictAvoidanceTable &CAT, bool isGoal) override;
};


#endif //ORCASTAR_SCIPP_H
