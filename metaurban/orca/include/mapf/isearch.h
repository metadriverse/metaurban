#include <list>
#include <unordered_set>
#include <queue>
#include <chrono>

#include "../geom.h"
#include "../sub_map.h"
#include "mapf_actor.h"
#include "mapf_actor_set.h"
#include "constraints.h"
#include "conflict_avoidance_table.h"
#include "search_queue.h"
#include "fs_node.h"
#include "scipp_node.h"

#ifndef ORCA_ISEARCH_H
#define ORCA_ISEARCH_H


struct SearchResult {
	bool pathfound;
	float pathlength; //if path not found, then pathlength=0
	std::list<Node> lppath; //path as the sequence of adjacent nodes (see above)
	//This is a pointer to the list of nodes that is actually created and hadled by ISearch class,
	//so no need to re-create them, delete them etc. It's just a trick to save some memory
	std::list<Node> hppath; //path as the sequence of non-adjacent nodes: "sections" (see above)
	//This is a pointer to the list of nodes that is actually created and hadled by ISearch class,
	//so no need to re-create them, delete them etc. It's just a trick to save some memory
	unsigned int nodescreated; //|OPEN| + |CLOSE| = total number of nodes saved in memory during search process.
	unsigned int nodesexpanded;
	unsigned int numberofsteps; //number of iterations (expansions) made by algorithm to find a solution
	double time; //runtime of the search algorithm (expanding nodes + reconstructing the path)
	Node lastNode;
	double minF;

	SearchResult() {
		pathfound = false;
		pathlength = 0;
		lppath = std::list<Node>();
		hppath = std::list<Node>();
		nodescreated = 0;
		numberofsteps = 0;
		time = 0;
	}

};

template<typename NodeType = Node>
class ISearch {
	public:
		ISearch(bool WithTime = false);

		virtual ~ISearch(void);

		SearchResult startSearch(const SubMap &map, const MAPFActorSet &agentSet,
								 int start_i, int start_j, int goal_i = 0, int goal_j = 0,
								 bool (*isGoal)(const Node &, const Node &, const SubMap &,
												const MAPFActorSet &) = nullptr,
								 bool freshStart = true, bool returnPath = true,
								 int startTime = 0, int goalTime = -1, int maxTime = -1,
								 const std::unordered_set<Node, NodeHash> &occupiedNodes =
								 std::unordered_set<Node, NodeHash>(),
								 const ConstraintsSet &constraints = ConstraintsSet(),
								 bool withCAT = false, const ConflictAvoidanceTable &CAT = ConflictAvoidanceTable());

		virtual std::list<NodeType>
		findSuccessors(const NodeType &curNode, const SubMap &map, int goal_i = 0, int goal_j = 0, int agentId = -1,
					   const std::unordered_set<Node, NodeHash> &occupiedNodes =
					   std::unordered_set<Node, NodeHash>(),
					   const ConstraintsSet &constraints = ConstraintsSet(),
					   bool withCAT = false, const ConflictAvoidanceTable &CAT = ConflictAvoidanceTable());

		//static int convolution(int i, int j, const Map &map, int time = 0, bool withTime = false);

		std::unordered_map<std::pair<NodeType, NodeType>, int, NodePairHash>
		getPerfectHeuristic(const SubMap &map, const MAPFActorSet &agentSet);

		// void getPerfectHeuristic(const Map &map, const AgentSet &agentSet);
		virtual double computeHFromCellToCell(int start_i, int start_j, int fin_i, int fin_j) { return 0; }

		static int T;

	protected:

		virtual void makePrimaryPath(Node &curNode, int endTime);//Makes path using back pointers
		virtual void makeSecondaryPath(const SubMap &map);//Makes another type of path(sections or points)
		virtual void setEndTime(NodeType &node, int start_i, int start_j, int startTime, int agentId,
								const ConstraintsSet &constraints);

		virtual void setHC(NodeType &neigh, const NodeType &cur,
						   const ConflictAvoidanceTable &CAT, bool isGoal) {}

		virtual void createSuccessorsFromNode(const NodeType &cur, NodeType &neigh, std::list<NodeType> &successors,
											  int agentId, const ConstraintsSet &constraints,
											  const ConflictAvoidanceTable &CAT, bool isGoal);

		virtual bool checkGoal(const NodeType &cur, int goalTime, int agentId, const ConstraintsSet &constraints);

		virtual void addStartNode(NodeType &node, const SubMap &map, const ConflictAvoidanceTable &CAT);

		virtual void addSuboptimalNode(NodeType &node, const SubMap &map, const ConflictAvoidanceTable &CAT) {}

		virtual bool checkOpenEmpty();

		virtual bool canStay() { return withTime; }

		virtual int getFocalSize() { return 0; }

		virtual NodeType getCur(const SubMap &map);

		virtual void subtractFutureConflicts(NodeType &node) {}

		virtual bool updateFocal(const NodeType &neigh, const SubMap &map);

		virtual double getMinFocalF();

		virtual void clearLists();

		SearchResult sresult;
		std::list<Node> lppath, hppath;
		double hweight;//weight of h-value
		bool breakingties;//flag that sets the priority of nodes in addOpen function when their F-values is equal
		SearchQueue<NodeType> open;
		std::unordered_map<int, NodeType> close;
		bool withTime;
		//need to define open, close;

};


#endif //ORCA_ISEARCH_H
