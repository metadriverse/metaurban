#include "isearch.h"


#ifndef ORCA_ASTAR_H
#define ORCA_ASTAR_H


template<typename NodeType = Node>
class Astar : public ISearch<NodeType> {
	public:
		Astar(bool WithTime = false, double HW = 1.0, bool BT = true);

		virtual ~Astar() {}

		double computeHFromCellToCell(int i1, int j1, int i2, int j2) override;

		void getPerfectHeuristic(const SubMap &map, const MAPFActorSet &agentSet);

	protected:
		double euclideanDistance(int x1, int y1, int x2, int y2);

		double manhattanDistance(int x1, int y1, int x2, int y2);

		double chebyshevDistance(int x1, int y1, int x2, int y2);

		double diagonalDistance(int x1, int y1, int x2, int y2);

		double metric(int x1, int y1, int x2, int y2);

		std::unordered_map<std::pair<Node, Node>, int, NodePairHash> perfectHeuristic;
};


#endif //ORCA_ASTAR_H
