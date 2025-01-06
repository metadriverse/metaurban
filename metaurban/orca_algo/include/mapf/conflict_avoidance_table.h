#include <unordered_map>
#include <list>
#include <map>
#include <unordered_map>

#include "../geom.h"
#include "../sub_map.h"

#ifndef ORCA_CONFLICTAVOIDANCETABLE_H
#define ORCA_CONFLICTAVOIDANCETABLE_H

class ConflictAvoidanceTable {
	public:
		void addNode(const Node &node);

		void addEdge(const Node &node, const Node &prev);

		void removeNode(const Node &node);

		void removeEdge(const Node &node, const Node &prev);

		void addAgentPath(const std::list<Node>::const_iterator &start,
						  const std::list<Node>::const_iterator &end);

		void removeAgentPath(const std::list<Node>::const_iterator &start,
							 const std::list<Node>::const_iterator &end);

		int getAgentsCount(const Node &node) const;

		int getFirstSoftConflict(const Node &node, int startTime, int endTime) const;

		int getFutureConflictsCount(const Node &node, int time) const;

		void getSoftConflictIntervals(std::vector<std::pair<int, int>> &res, const Node &node, const Node &prevNode,
									  int startTime, int endTime, bool binary) const;

	private:
		std::map<std::tuple<int, int, int>, int> nodeAgentsCount;
		std::map<std::tuple<int, int, int, int, int>, int> edgeAgentsCount;
};


#endif //ORCA_CONFLICTAVOIDANCETABLE_H
