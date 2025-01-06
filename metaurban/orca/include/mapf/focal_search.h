#ifndef ORCASTAR_FOCALSEARCH_H
#define ORCASTAR_FOCALSEARCH_H

#include "astar.h"
#include "fs_node.h"

#ifndef FOCALSEARCH_H
#define FOCALSEARCH_H


template<typename NodeType = FSNode>
class FocalSearch : virtual public Astar<NodeType> {
	public:
		FocalSearch(bool WithTime = false, double FocalW = 1.0, double HW = 1.0, bool BT = true);

		virtual ~FocalSearch() {}

	protected:
		bool checkOpenEmpty() override;

		NodeType getCur(const SubMap &map) override;

		bool updateFocal(const NodeType &neigh, const SubMap &map) override;

		double getMinFocalF() override;

		void clearLists() override;

		virtual void setHC(NodeType &neigh, const NodeType &cur,
						   const ConflictAvoidanceTable &CAT, bool isGoal) override;

		virtual int getFocalSize() override { return focal.size(); }

		virtual void subtractFutureConflicts(NodeType &node) override { node.hc -= node.futureConflictsCount; }

		virtual void addFutureConflicts(NodeType &neigh, const ConflictAvoidanceTable &CAT);

		SearchQueue<NodeType> focal;
		std::multiset<double> focalF;
		double focalW;
};

#endif // FOCALSEARCH_H


#endif //ORCASTAR_FOCALSEARCH_H
