#include "mapf/isearch.h"

template<typename NodeType>
ISearch<NodeType>::ISearch(bool WithTime) {
	hweight = 1;
	breakingties = CN_SP_BT_GMAX;
	withTime = WithTime;
}

template<typename NodeType>
ISearch<NodeType>::~ISearch(void) {}

template<typename NodeType>
int ISearch<NodeType>::T = 0;

template<typename NodeType>
SearchResult ISearch<NodeType>::startSearch(const SubMap &map, const MAPFActorSet &agentSet,
											int start_i, int start_j, int goal_i, int goal_j,
											bool (*isGoal)(const Node &, const Node &, const SubMap &,
														   const MAPFActorSet &),
											bool freshStart, bool returnPath, int startTime, int goalTime, int maxTime,
											const std::unordered_set<Node, NodeHash> &occupiedNodes,
											const ConstraintsSet &constraints,
											bool withCAT, const ConflictAvoidanceTable &CAT) {
	sresult.pathfound = false;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	if (goalTime != -1) {
		maxTime = goalTime;
	}
	NodeType cur;
	int agentId = -1;
	if (agentSet.isOccupied(start_i, start_j)) {
		agentId = agentSet.getActorId(start_i, start_j);
	}

	if (withCAT) {
		open = SearchQueue<NodeType>([](const NodeType &lhs, const NodeType &rhs) {
			return std::tuple<int, int, int, int, int>(lhs.F, lhs.conflictsCount, -lhs.g, lhs.i, lhs.j) <
				   std::tuple<int, int, int, int, int>(rhs.F, rhs.conflictsCount, -rhs.g, rhs.i, rhs.j);
		});
	}

	if (freshStart) {
		clearLists();
		sresult.numberofsteps = 0;
		cur = NodeType(start_i, start_j, nullptr, startTime,
					   computeHFromCellToCell(start_i, start_j, goal_i, goal_j));
		setEndTime(cur, start_i, start_j, startTime, agentId, constraints);
		addStartNode(cur, map, CAT);
		addSuboptimalNode(cur, map, CAT);
	}

	while (!checkOpenEmpty()) {
		++sresult.numberofsteps;
		if (sresult.numberofsteps % 100000 == 0) {
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			int elapsedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
			if (elapsedMilliseconds > 10000) {
				break;
			}
		}

		cur = getCur(map);
		close[cur.convolution(map.GetWidth(), map.GetHeight(), withTime)] = cur;

		NodeType *curPtr = &(close.find(cur.convolution(map.GetWidth(), map.GetHeight(), withTime))->second);
		if ((isGoal != nullptr && isGoal(NodeType(start_i, start_j), cur, map, agentSet)) ||
			(isGoal == nullptr && cur.i == goal_i && cur.j == goal_j)) {
			if (!constraints.hasFutureConstraint(cur.i, cur.j, cur.g, agentId) &&
				checkGoal(cur, goalTime, agentId, constraints)) {
				if (!freshStart) {
					freshStart = true;
				}
				else {
					sresult.pathfound = true;
					break;
				}
			}
			else {
				subtractFutureConflicts(cur);
			}
		}

		if (maxTime == -1 || cur.g < maxTime) {
			std::list<NodeType> successors = findSuccessors(cur, map, goal_i, goal_j, agentId, occupiedNodes,
															constraints, withCAT, CAT);
			for (auto neigh: successors) {
				if (close.find(neigh.convolution(map.GetWidth(), map.GetHeight(), withTime)) == close.end()) {
					neigh.parent = curPtr;
					if (!updateFocal(neigh, map)) {
						open.insert(map, neigh, withTime);
					}
				}
			}
		}
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	int elapsedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	T += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

	sresult.time = static_cast<double>(elapsedMilliseconds) / 1000;
	sresult.nodescreated = open.size() + close.size() + getFocalSize();
	sresult.nodesexpanded = close.size();

	if (sresult.pathfound) {
		sresult.pathlength = cur.g;
		sresult.minF = std::min(double(cur.F), getMinFocalF());
		sresult.lastNode = cur;
		if (returnPath) {
			lppath.clear();
			hppath.clear();
			makePrimaryPath(cur, goalTime == -1 ? -1 : goalTime + 1);
			makeSecondaryPath(map);
			sresult.hppath = hppath; //Here is a constant pointer
			sresult.lppath = lppath;
		}
	}
	return sresult;
}

template<typename NodeType>
std::list<NodeType> ISearch<NodeType>::findSuccessors(const NodeType &curNode, const SubMap &map,
													  int goal_i, int goal_j, int agentId,
													  const std::unordered_set<Node, NodeHash> &occupiedNodes,
													  const ConstraintsSet &constraints,
													  bool withCAT, const ConflictAvoidanceTable &CAT) {
	std::list<NodeType> successors;
	for (int di = -1; di <= 1; ++di) {
		for (int dj = -1; dj <= 1; ++dj) {
			int newi = curNode.i + di, newj = curNode.j + dj;
			if ((di == 0 || dj == 0) && (canStay() || di != 0 || dj != 0) && map.CellOnGrid(newi, newj) &&
				map.CellIsTraversable(newi, newj, occupiedNodes)) {
				int newh = computeHFromCellToCell(newi, newj, goal_i, goal_j);
				NodeType neigh(newi, newj, nullptr, curNode.g + 1, newh);
				neigh.conflictsCount = CAT.getAgentsCount(neigh);
				createSuccessorsFromNode(curNode, neigh, successors, agentId, constraints, CAT,
										 neigh.i == goal_i && neigh.j == goal_j);
			}
		}
	}
	return successors;
}

template<typename NodeType>
void ISearch<NodeType>::clearLists() {
	open.clear();
	close.clear();
}

template<typename NodeType>
void ISearch<NodeType>::addStartNode(NodeType &node, const SubMap &map, const ConflictAvoidanceTable &CAT) {
	open.insert(map, node, withTime);
}

template<typename NodeType>
bool ISearch<NodeType>::checkOpenEmpty() {
	return open.empty();
}

template<typename NodeType>
NodeType ISearch<NodeType>::getCur(const SubMap &map) {
	NodeType cur = open.getFront();
	open.erase(map, cur, withTime);
	return cur;
}

template<typename NodeType>
bool ISearch<NodeType>::updateFocal(const NodeType &neigh, const SubMap &map) {
	return false;
}

template<typename NodeType>
double ISearch<NodeType>::getMinFocalF() {
	return CN_INFINITY;
}

template<typename NodeType>
void ISearch<NodeType>::setEndTime(NodeType &node, int start_i, int start_j, int startTime, int agentId,
								   const ConstraintsSet &constraints) {
	return;
}

template<typename NodeType>
bool ISearch<NodeType>::checkGoal(const NodeType &cur, int goalTime, int agentId, const ConstraintsSet &constraints) {
	return goalTime == -1 || cur.g == goalTime;
}

template<typename NodeType>
void ISearch<NodeType>::createSuccessorsFromNode(const NodeType &cur, NodeType &neigh, std::list<NodeType> &successors,
												 int agentId, const ConstraintsSet &constraints,
												 const ConflictAvoidanceTable &CAT, bool isGoal) {
	if (!constraints.hasNodeConstraint(neigh.i, neigh.j, neigh.g, agentId) &&
		!constraints.hasEdgeConstraint(neigh.i, neigh.j, neigh.g, agentId, cur.i, cur.j)) {
		setHC(neigh, cur, CAT, isGoal);
		successors.push_back(neigh);
	}
}

template<typename NodeType>
void ISearch<NodeType>::makePrimaryPath(Node &curNode, int endTime) {
	if (withTime && endTime != -1) {
		int startTime = curNode.g;
		for (curNode.g = endTime - 1; curNode.g > startTime; --curNode.g) {
			lppath.push_front(curNode);
		}
	}
	lppath.push_front(curNode);
	if (curNode.parent != nullptr) {
		makePrimaryPath(*(curNode.parent), curNode.g);
	}
}

template<typename NodeType>
void ISearch<NodeType>::makeSecondaryPath(const SubMap &map) {
	auto it = lppath.begin();
	hppath.push_back(*it);
	++it;
	for (it; it != lppath.end(); ++it) {
		auto prevIt = it;
		--prevIt;
		auto nextIt = it;
		++nextIt;
		if (nextIt == lppath.end() ||
			(it->i - prevIt->i) * (nextIt->j - it->j) != (it->j - prevIt->j) * (nextIt->i - it->i)) {
			hppath.push_back(*it);
		}
	}
}

template
class ISearch<Node>;

template
class ISearch<FSNode>;

template
class ISearch<SIPPNode>;

template
class ISearch<SCIPPNode>;
