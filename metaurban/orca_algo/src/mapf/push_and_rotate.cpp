
#include "mapf/push_and_rotate.h"

PushAndRotate::PushAndRotate() {
	search = nullptr;
}

PushAndRotate::PushAndRotate(ISearch<> *Search) {
	search = Search;
}

PushAndRotate::~PushAndRotate() {
//    if (search)
//        delete search;
}

void PushAndRotate::clear() {
	agentsPaths.clear();
	agentsMoves.clear();
}

bool PushAndRotate::clearNode(const SubMap &map, MAPFActorSet &agentSet, Node &nodeToClear,
							  const std::unordered_set<Node, NodeHash> &occupiedNodes) {
	auto isGoal = [](const Node &start, const Node &cur, const SubMap &map, const MAPFActorSet &agentSet) {
		return !agentSet.isOccupied(cur.i, cur.j);
	};

	ISearch<> dijkstraSearch;
	SearchResult searchResult = dijkstraSearch.startSearch(map, agentSet, nodeToClear.i, nodeToClear.j, 0, 0,
														   isGoal, true, true, 0, -1, -1, occupiedNodes);
	if (!searchResult.pathfound) {
		return false;
	}
	auto path = searchResult.lppath;
	for (auto it = std::next(path.rbegin()); it != path.rend(); ++it) {
		if (agentSet.isOccupied(it->i, it->j)) {
			Node from = *it;
			Node to = *std::prev(it);
			agentSet.moveActor(from, to, agentsMoves);
		}
	}
	return true;
}

bool PushAndRotate::push(const SubMap &map, MAPFActorSet &agentSet, Node &from, Node &to,
						 std::unordered_set<Node, NodeHash> &occupiedNodes) {
	if (occupiedNodes.find(to) != occupiedNodes.end()) {
		return false;
	}
	if (agentSet.isOccupied(to.i, to.j)) {
		bool inserted = false;
		if (occupiedNodes.find(from) == occupiedNodes.end()) {
			occupiedNodes.insert(from);
			inserted = true;
		}
		bool canClear = clearNode(map, agentSet, to, occupiedNodes);
		if (inserted) {
			occupiedNodes.erase(from);
		}
		if (!canClear) {
			return false;
		}
	}
	agentSet.moveActor(from, to, agentsMoves);
	return true;
}

bool PushAndRotate::multipush(const SubMap &map, MAPFActorSet &agentSet, Node first, Node second, Node &to,
							  std::list<Node> &path) {
	if (path.size() > 1 && *std::next(path.begin()) == second) {
		std::swap(first, second);
		path.pop_front();
	}
	Node prevNode = second;
	for (auto it = path.begin(); it != std::prev(path.end()); ++it) {
		Node curNode = *it;
		Node nextNode = *std::next(it);
		std::unordered_set<Node, NodeHash> occupiedNodes = {prevNode, curNode};
		if (agentSet.isOccupied(nextNode.i, nextNode.j)) {
			if (!clearNode(map, agentSet, nextNode, occupiedNodes)) {
				return false;
			}
		}
		agentSet.moveActor(curNode, nextNode, agentsMoves);
		agentSet.moveActor(prevNode, curNode, agentsMoves);
		prevNode = curNode;
	}
	return true;
}

bool PushAndRotate::clear(const SubMap &map, MAPFActorSet &agentSet, Node &first, Node &second) {
	std::list<Node> successors = search->findSuccessors(first, map);
	std::set<Node> unoccupied;
	for (auto node: successors) {
		if (!agentSet.isOccupied(node.i, node.j)) {
			unoccupied.insert(node);
		}
	}
	if (unoccupied.size() >= 2) {
		return true;
	}

	std::unordered_set<Node, NodeHash> forbidden = {first, second};
	forbidden.insert(unoccupied.begin(), unoccupied.end());
	for (auto node: successors) {
		if (unoccupied.find(node) == unoccupied.end() && node != second &&
			clearNode(map, agentSet, node, forbidden)) {
			if (unoccupied.size() >= 1) {
				return true;
			}
			unoccupied.insert(node);
			forbidden.insert(node);
		}
	}
	if (unoccupied.empty()) {
		return false;
	}

	Node freeNeigh = *unoccupied.begin();
	for (auto node: successors) {
		if (node != second && node != freeNeigh) {
			size_t curSize = agentsMoves.size();
			MAPFActorSet newAgentSet = agentSet;
			if (clearNode(map, newAgentSet, node, {first, second})) {
				if (clearNode(map, newAgentSet, freeNeigh, {first, second, node})) {
					agentSet = newAgentSet;
					return true;
				}
			}
			agentsMoves.erase(agentsMoves.begin() + curSize, agentsMoves.end());
			break;
		}
	}

	for (auto node: successors) {
		if (node != second && node != freeNeigh) {
			int curSize = agentsMoves.size();
			MAPFActorSet newAgentSet = agentSet;
			newAgentSet.moveActor(first, freeNeigh, agentsMoves);
			newAgentSet.moveActor(second, first, agentsMoves);
			if (clearNode(map, newAgentSet, node, {first, second})) {
				if (clearNode(map, newAgentSet, second, {first, second, node})) {
					agentSet = newAgentSet;
					return true;
				}
			}
			agentsMoves.erase(agentsMoves.begin() + curSize, agentsMoves.end());
			break;
		}
	}

	int secondAgentId = agentSet.getActorId(second.i, second.j);
	if (!clearNode(map, agentSet, second, {first})) {
		return false;
	}
	agentSet.moveActor(first, second, agentsMoves);
	Node secondPosition = agentSet.getActor(secondAgentId).getCurPosition();
	if (!clearNode(map, agentSet, freeNeigh, {first, second, secondPosition})) {
		return false;
	}
	for (auto node: successors) {
		if (node != second && node != freeNeigh) {
			agentSet.moveActor(node, first, agentsMoves);
			agentSet.moveActor(first, freeNeigh, agentsMoves);
			agentSet.moveActor(second, first, agentsMoves);
			agentSet.moveActor(secondPosition, second, agentsMoves);
			return clearNode(map, agentSet, freeNeigh, {first, second, node});
		}
	}
	return false;
}

void PushAndRotate::exchange(const SubMap &map, MAPFActorSet &agentSet, Node &first, Node &second) {
	std::list<Node> successors = search->findSuccessors(first, map);
	std::vector<Node> freeNeigh;
	for (auto node: successors) {
		if (!agentSet.isOccupied(node.i, node.j)) {
			freeNeigh.push_back(node);
		}
	}
	agentSet.moveActor(first, freeNeigh[0], agentsMoves);
	agentSet.moveActor(second, first, agentsMoves);
	agentSet.moveActor(first, freeNeigh[1], agentsMoves);
	agentSet.moveActor(freeNeigh[0], first, agentsMoves);
	agentSet.moveActor(first, second, agentsMoves);
	agentSet.moveActor(freeNeigh[1], first, agentsMoves);
}

void PushAndRotate::reverse(int begSize, int endSize,
							int firstAgentId, int secondAgentId, MAPFActorSet &agentSet) {
	for (int i = endSize - 1; i >= begSize; --i) {
		ActorMove pos = agentsMoves[i];
		if (pos.id == firstAgentId) {
			pos.id = secondAgentId;
		}
		else if (pos.id == secondAgentId) {
			pos.id = firstAgentId;
		}
		Node from = agentSet.getActor(pos.id).getCurPosition();
		Node to = Node(from.i - pos.di, from.j - pos.dj);
		agentSet.moveActor(from, to, agentsMoves);
	}
}

bool PushAndRotate::swap(const SubMap &map, MAPFActorSet &agentSet, Node &first, Node &second) {
	int firstAgentId = agentSet.getActorId(first.i, first.j);
	int secondAgentId = agentSet.getActorId(second.i, second.j);

	auto isGoal = [](const Node &start, const Node &cur, const SubMap &map, const MAPFActorSet &agentSet) {
		return map.GetCellDegree(cur.i, cur.j) >= 3;
	};

	ISearch<> dijkstraSearch;
	SearchResult searchResult = dijkstraSearch.startSearch(map, agentSet, first.i, first.j, 0, 0, isGoal);
	while (searchResult.pathfound) {
		int begSize = agentsMoves.size();
		MAPFActorSet newAgentSet = agentSet;
		auto path = searchResult.lppath;
		Node exchangeNode = path.back();
		if (multipush(map, newAgentSet, first, second, exchangeNode, path)) {
			int exchangeAgentId = newAgentSet.getActorId(exchangeNode.i, exchangeNode.j);
			int neighAgentId = (exchangeAgentId == firstAgentId) ? secondAgentId : firstAgentId;
			Node neigh = newAgentSet.getActor(neighAgentId).getCurPosition();
			if (clear(map, newAgentSet, exchangeNode, neigh)) {
				agentSet = newAgentSet;
				int endSize = agentsMoves.size();
				exchange(map, agentSet, exchangeNode, neigh);
				reverse(begSize, endSize, firstAgentId, secondAgentId, agentSet);
				return true;
			}
		}
		searchResult = dijkstraSearch.startSearch(map, agentSet, first.i, first.j, 0, 0, isGoal, false);
	}
	return false;
}

bool PushAndRotate::rotate(const SubMap &map, MAPFActorSet &agentSet, std::vector<Node> &qPath, int cycleBeg) {
	int size = qPath.size() - cycleBeg;
	for (int i = cycleBeg; i < qPath.size(); ++i) {
		if (!agentSet.isOccupied(qPath[i].i, qPath[i].j)) {
			for (int j = 0; j < size - 1; ++j) {
				int from = cycleBeg + (i - cycleBeg - j - 1 + size) % size;
				int to = cycleBeg + (i - cycleBeg - j + size) % size;
				if (agentSet.isOccupied(qPath[from].i, qPath[from].j)) {
					agentSet.moveActor(qPath[from], qPath[to], agentsMoves);
				}
			}
			return true;
		}
	}

	std::unordered_set<Node, NodeHash> cycleNodes(qPath.begin() + cycleBeg, qPath.end());
	for (int i = cycleBeg; i < qPath.size(); ++i) {
		cycleNodes.erase(qPath[i]);
		int firstAgentId = agentSet.getActorId(qPath[i].i, qPath[i].j);
		int begSize = agentsMoves.size();
		if (clearNode(map, agentSet, qPath[i], cycleNodes)) {
			int endSize = agentsMoves.size();
			int secondAgentIndex = cycleBeg + (i - cycleBeg - 1 + size) % size;
			int secondAgentId = agentSet.getActorId(qPath[secondAgentIndex].i, qPath[secondAgentIndex].j);
			agentSet.moveActor(qPath[secondAgentIndex], qPath[i], agentsMoves);
			Node curPosition = agentSet.getActor(firstAgentId).getCurPosition();
			swap(map, agentSet, qPath[i], curPosition);
			for (int j = 0; j < size - 1; ++j) {
				int from = cycleBeg + (i - cycleBeg - j - 2 + size) % size;
				int to = cycleBeg + (i - cycleBeg - j - 1 + size) % size;
				if (agentSet.isOccupied(qPath[from].i, qPath[from].j)) {
					agentSet.moveActor(qPath[from], qPath[to], agentsMoves);
				}
			}
			reverse(begSize, endSize, firstAgentId, secondAgentId, agentSet);
			return true;
		}
		cycleNodes.insert(qPath[i]);
	}
	return false;
}

void PushAndRotate::getPaths(MAPFActorSet &agentSet) {
	agentsPaths.resize(agentSet.getActorCount());
	std::vector<Node> agentPositions;
	for (int i = 0; i < agentSet.getActorCount(); ++i) {
		Node startPosition = agentSet.getActor(i).getStartPosition();
		agentPositions.push_back(startPosition);
		agentsPaths[i].push_back(startPosition);
	}
	for (int i = 0; i < agentsMoves.size(); ++i) {
		agentPositions[agentsMoves[i].id].i += agentsMoves[i].di;
		agentPositions[agentsMoves[i].id].j += agentsMoves[i].dj;
		for (int j = 0; j < agentSet.getActorCount(); ++j) {
			agentsPaths[j].push_back(agentPositions[j]);
		}
	}
}

void PushAndRotate::getParallelPaths(MAPFActorSet &agentSet, const MAPFConfig &config) {
	int agentCount = agentSet.getActorCount();
	std::vector<std::vector<Node>> agentsPositions(agentCount);
	std::vector<int> agentInd(agentCount, 0);
	std::unordered_map<Node, std::vector<int>, NodeHash> nodesOccupations;
	std::unordered_map<Node, int, NodeHash> nodeInd;

	agentsPaths.resize(agentSet.getActorCount());
	for (int i = 0; i < agentSet.getActorCount(); ++i) {
		Node startPosition = agentSet.getActor(i).getStartPosition();
		agentsPositions[i].push_back(startPosition);
		agentsPaths[i].push_back(startPosition);
		if (nodesOccupations.find(startPosition) == nodesOccupations.end()) {
			nodesOccupations[startPosition] = {};
			nodeInd[startPosition] = 0;
		}
		nodesOccupations[startPosition].push_back(i);
	}

	for (auto move: agentsMoves) {
		Node cur = agentsPositions[move.id].back();
		cur.i += move.di;
		cur.j += move.dj;
		if (nodesOccupations.find(cur) == nodesOccupations.end()) {
			nodesOccupations[cur] = {};
			nodeInd[cur] = 0;
		}
		if (!nodesOccupations[cur].empty() && nodesOccupations[cur].back() == move.id) {
			while (agentsPositions[move.id].back() != cur) {
				Node curBack = agentsPositions[move.id].back();
				int lastInd;
				for (lastInd = nodesOccupations[curBack].size() - 1;
					 lastInd >= 0 && nodesOccupations[curBack][lastInd] != move.id; --lastInd);
				nodesOccupations[curBack].erase(nodesOccupations[curBack].begin() + lastInd);
				agentsPositions[move.id].pop_back();
			}
		}
		else {
			agentsPositions[move.id].push_back(cur);
			nodesOccupations[cur].push_back(move.id);
		}
	}

	std::vector<bool> finished(agentCount, false);
	while (true) {
		std::vector<bool> hasMoved(agentCount, false);
		for (int i = 0; i < agentCount; ++i) {
			if (hasMoved[i] || finished[i]) {
				continue;
			}
			if (agentsPositions[i].size() == 1) {
				agentsPaths[i].push_back(agentsPositions[i][0]);
				finished[i] = true;
				continue;
			}

			std::vector<int> path;
			int curAgent = i;
			bool canMove = true;
			while (true) {
				path.push_back(curAgent);
				Node nextNode = agentsPositions[curAgent][agentInd[curAgent] + 1];
				int lastInd = nodeInd[nextNode];
				if (nodesOccupations[nextNode][lastInd] == curAgent) {
					break;
				}
				else if (nodesOccupations[nextNode][lastInd + 1] == curAgent) {
					int nextAgent = nodesOccupations[nextNode][lastInd];
					if (finished[nextAgent] || hasMoved[nextAgent] || nextAgent < curAgent ||
						agentsPositions[nextAgent][agentInd[nextAgent]] != nextNode) {
						canMove = false;
						break;
					}
					curAgent = nextAgent;
					if (curAgent == i) {
						break;
					}
				}
				else {
					canMove = false;
					break;
				}
			}

			if (canMove) {
				for (int agentId: path) {
					hasMoved[agentId] = true;
					++nodeInd[agentsPositions[agentId][agentInd[agentId]]];
					++agentInd[agentId];
					agentsPaths[agentId].push_back(agentsPositions[agentId][agentInd[agentId]]);
					if (agentInd[agentId] == agentsPositions[agentId].size() - 1) {
						finished[agentId] = true;
					}
				}
			}
			else {
				agentsPaths[i].push_back(agentsPositions[i][agentInd[i]]);
			}
		}

		if (!std::any_of(hasMoved.begin(), hasMoved.end(), [](const bool x) { return x; })) {
			break;
		}
	}

	if (config.parallelizePaths2) {
		ConstraintsSet constraints;
		for (int i = 0; i < agentCount; ++i) {
			constraints.addAgentPath<std::vector<Node>::iterator>(agentsPaths[i].begin(), agentsPaths[i].end(), i);
		}
		for (int i = 0; i < agentCount; ++i) {
			constraints.removeAgentPath<std::vector<Node>::iterator>(agentsPaths[i].begin(), agentsPaths[i].end(), i);
			std::vector<std::vector<bool>> dp(agentsPaths[i].size()), move(agentsPaths[i].size());
			dp[0] = {true};
			move[0] = {false};
			int last = agentsPositions[i].size() - 2;
			for (size_t time = 1; time < agentsPaths[i].size(); ++time) {
				for (int j = 0; j <= time && j < agentsPositions[i].size(); ++j) {
					dp[time].push_back(false);
					move[time].push_back(false);
					Node curPos = agentsPositions[i][j];
					if (!constraints.hasNodeConstraint(curPos.i, curPos.j, time, i) &&
						(j < agentsPositions[i].size() - 1 ||
						 !constraints.hasFutureConstraint(curPos.i, curPos.j, time, i))) {
						if (j < time && dp[time - 1][j]) {
							dp[time][j] = true;
						}
						else if (j > 0) {
							Node prevPos = agentsPositions[i][j - 1];
							if (dp[time - 1][j - 1] &&
								!constraints.hasEdgeConstraint(curPos.i, curPos.j, time, i, prevPos.i, prevPos.j)) {
								dp[time][j] = true;
								move[time][j] = true;
							}
						}
					}
				}

				if (dp[time].size() >= agentsPositions[i].size() && !dp[time][agentsPositions[i].size() - 1]) {
					last = time;
				}
			}

			agentsPaths[i].clear();
			int posInd = agentsPositions[i].size() - 1;
			for (int j = last + 1; j >= 0; --j) {
				agentsPaths[i].push_back(agentsPositions[i][posInd]);
				if (move[j][posInd]) {
					--posInd;
				}
			}
			std::reverse(agentsPaths[i].begin(), agentsPaths[i].end());
			constraints.addAgentPath<std::vector<Node>::iterator>(agentsPaths[i].begin(), agentsPaths[i].end(), i);
		}
	}
}

bool PushAndRotate::solve(const SubMap &map, const MAPFConfig &config, MAPFActorSet &agentSet,
						  std::chrono::steady_clock::time_point begin) {
	auto comparator = [&agentSet](int id1, int id2) {
		int subgraph1 = agentSet.getActor(id1).getSubgraph();
		int subgraph2 = agentSet.getActor(id2).getSubgraph();

		if (subgraph1 != subgraph2) {
			if (subgraph1 == -1 || agentSet.hasPriority(subgraph2, subgraph1)) {
				return false;
			}
			else if (subgraph2 == -1 || agentSet.hasPriority(subgraph1, subgraph2)) {
				return true;
			}
		}
		return id1 < id2;
	};

	std::set<int, decltype(comparator)> notFinished(comparator);
	std::unordered_set<int> finished;
	std::unordered_set<Node, NodeHash> qPathNodes, finishedPositions;
	std::vector<Node> qPath;

	for (int i = 0; i < agentSet.getActorCount(); ++i) {
		notFinished.insert(i);
	}

	bool isPolygon = true;
	for (int i = 0; i < map.GetHeight(); ++i) {
		for (int j = 0; j < map.GetWidth(); ++j) {
			if (!map.CellIsObstacle(i, j) && map.GetCellDegree(i, j) != 2) {
				isPolygon = false;
				break;
			}
		}
		if (!isPolygon) {
			break;
		}
	}

	int curAgentId = -1;
	MAPFActor curAgent;
	while (!notFinished.empty()) {
		std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
		if (std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count() > config.maxTime) {
			return false;
		}

		if (curAgentId == -1) {
			curAgent = agentSet.getActor(*notFinished.begin());
		}
		else {
			curAgent = agentSet.getActor(curAgentId);
		}
		notFinished.erase(curAgent.getId());

		SearchResult searchResult = search->startSearch(map, agentSet,
														curAgent.getCur_i(), curAgent.getCur_j(),
														curAgent.getGoal_i(), curAgent.getGoal_j(),
														nullptr, true, true, 0, -1, -1,
														isPolygon ? finishedPositions
																  : std::unordered_set<Node, NodeHash>());

		if (!searchResult.pathfound) {
			return false;
		}

		auto path = searchResult.lppath;
		qPath.push_back(*path.begin());
		qPathNodes.insert(*path.begin());
		for (auto it = path.begin(); it != std::prev(path.end()); ++it) {
			if (qPathNodes.find(*std::next(it)) != qPathNodes.end()) {
				int cycleBeg;
				for (cycleBeg = qPath.size() - 1; cycleBeg >= 0 && qPath[cycleBeg] != *std::next(it); --cycleBeg);
				rotate(map, agentSet, qPath, cycleBeg);

				bool toErase = false;
				while (qPath.size() != cycleBeg) {
					Node lastNode = qPath.back();
					if (agentSet.isOccupied(lastNode.i, lastNode.j) &&
						finished.find(agentSet.getActorId(lastNode.i, lastNode.j)) != finished.end()) {
						if (!toErase) {
							finishedPositions.insert(lastNode);
							toErase = true;
						}
					}
					else {
						if (toErase) {
							finishedPositions.erase(lastNode);
							toErase = false;
						}
					}
					qPathNodes.erase(lastNode);
					qPath.pop_back();
				}
			}
			else if (!push(map, agentSet, *it, *std::next(it), finishedPositions)) {
				if (!swap(map, agentSet, *it, *std::next(it))) {
					return false;
				}
				if (finished.find(agentSet.getActorId(it->i, it->j)) != finished.end()) {
					finishedPositions.erase(*std::next(it));
					finishedPositions.insert(*it);
				}
			}
			qPath.push_back(*std::next(it));
			qPathNodes.insert(*std::next(it));
		}
		finished.insert(curAgent.getId());
		finishedPositions.insert(curAgent.getGoalPosition());

		curAgentId = -1;
		while (!qPath.empty()) {
			Node lastNode = qPath.back();
			if (agentSet.isOccupied(lastNode.i, lastNode.j)) {
				MAPFActor curAgent = agentSet.getActor(agentSet.getActorId(lastNode.i, lastNode.j));
				Node goal = Node(curAgent.getGoal_i(), curAgent.getGoal_j());
				if (notFinished.find(curAgent.getId()) == notFinished.end() && lastNode != goal) {
					if (!agentSet.isOccupied(goal.i, goal.j)) {
						agentSet.moveActor(lastNode, goal, agentsMoves);
						finishedPositions.erase(lastNode);
						finishedPositions.insert(goal);
					}
					else {
						curAgentId = agentSet.getActorId(goal.i, goal.j);
						break;
					}
				}
			}
			qPathNodes.erase(lastNode);
			qPath.pop_back();
		}
	}
	return true;
}

void PushAndRotate::getComponent(MAPFActorSet &agentSet, std::pair<Node, Node> &startEdge,
								 std::vector<std::pair<Node, Node>> &edgeStack,
								 std::vector<std::unordered_set<Node, NodeHash>> &components) {
	std::unordered_set<Node, NodeHash> component;
	std::pair<Node, Node> curEdge;
	do {
		curEdge = edgeStack.back();
		component.insert(curEdge.first);
		component.insert(curEdge.second);
		edgeStack.pop_back();
	} while (curEdge != startEdge);
	if (component.size() <= 2) {
		return;
	}
	for (auto node: component) {
		agentSet.setNodeSubgraph(node.i, node.j, components.size());
	}
	components.push_back(component);

}

void
PushAndRotate::combineNodeSubgraphs(MAPFActorSet &agentSet, std::vector<std::unordered_set<Node, NodeHash>> &components,
									Node &subgraphNode, int subgraphNum) {
	std::vector<int> subgraphs = agentSet.getSubgraphs(subgraphNode.i, subgraphNode.j);
	for (int j = 0; j < subgraphs.size(); ++j) {
		if (subgraphs[j] != subgraphNum) {
			for (auto node: components[subgraphs[j]]) {
				agentSet.removeSubgraphs(node.i, node.j);
				agentSet.setNodeSubgraph(node.i, node.j, subgraphNum);
			}
			components[subgraphs[j]].clear();
		}
	}
}

void PushAndRotate::getSubgraphs(const SubMap &map, MAPFActorSet &agentSet) {
	std::unordered_set<Node, NodeHash> close;
	std::vector<std::unordered_set<Node, NodeHash>> components;
	std::unordered_set<Node, NodeHash> joinNodes;
	int connectedComponentNum = 0;
	for (int i = 0; i < map.GetHeight(); ++i) {
		for (int j = 0; j < map.GetWidth(); ++j) {
			if (!map.CellIsObstacle(i, j)) {
				Node curNode = Node(i, j);
				if (close.find(curNode) == close.end()) {
					int oldSize = close.size();
					std::vector<std::pair<Node, Node>> edgeStack;
					std::unordered_map<Node, int, NodeHash> in, up;
					std::vector<std::tuple<Node, int, int>> stack = {std::make_tuple(curNode, -1, 0)};

					while (!stack.empty()) {
						std::tuple<Node, int, int> state = stack.back();
						Node cur = std::get<0>(state);
						int lastInd = std::get<1>(state);
						int depth = std::get<2>(state);

						std::list<Node> successors = search->findSuccessors(cur, map);
						std::list<Node>::iterator it = successors.begin();
						for (int i = 0; i < lastInd; ++i, ++it) {}
						if (lastInd == -1) {
							close.insert(cur);
							agentSet.setConnectedComponent(cur.i, cur.j, connectedComponentNum);
							in[cur] = depth;
							up[cur] = depth;
						}
						else {
							if ((depth != 0 && up[*it] >= in[cur]) || depth == 0) {
								std::pair<Node, Node> curEdge = std::make_pair(cur, *it);
								getComponent(agentSet, curEdge, edgeStack, components);
								if (depth != 0) {
									joinNodes.insert(cur);
								}
							}
							up[cur] = std::min(up[*it], up[cur]);
							it = std::next(it);
						}
						for (it, lastInd = lastInd + 1; it != successors.end(); ++it, ++lastInd) {
							if (close.find(*it) != close.end()) {
								up[cur] = std::min(in[*it], up[cur]);
							}
							else {
								std::pair<Node, Node> curEdge = std::make_pair(cur, *it);
								edgeStack.push_back(curEdge);
								std::get<1>(stack.back()) = lastInd;
								stack.push_back(std::make_tuple(*it, -1, depth + 1));
								break;
							}
						}
						if (it == successors.end()) {
							stack.pop_back();
						}
					}

					agentSet.addComponentSize(close.size() - oldSize);
					++connectedComponentNum;
				}
			}
		}
	}

	for (int i = 0; i < map.GetHeight(); ++i) {
		for (int j = 0; j < map.GetWidth(); ++j) {
			if (!map.CellIsObstacle(i, j) && map.GetCellDegree(i, j) >= 3 && agentSet.getSubgraphs(i, j).empty()) {
				agentSet.setNodeSubgraph(i, j, components.size());
				components.push_back({Node(i, j)});
				joinNodes.insert(Node(i, j));
			}
		}
	}

	int m = map.GetEmptyCellCount() - agentSet.getActorCount();
	auto isGoal = [](const Node &start, const Node &cur, const SubMap &map, const MAPFActorSet &agentSet) {
		std::vector<int> startSubgraphs = agentSet.getSubgraphs(start.i, start.j);
		std::vector<int> curSubgraphs = agentSet.getSubgraphs(cur.i, cur.j);
		return curSubgraphs.size() > 1 || curSubgraphs.size() == 1 && curSubgraphs[0] != startSubgraphs[0];
	};

	std::vector<int> order;
	for (int i = 0; i < components.size(); ++i) {
		order.push_back(i);
	}
	std::sort(order.begin(), order.end(),
			  [&components](const int a, const int b) { return components[a].size() > components[b].size(); });

	for (int i: order) {
		for (auto start: components[i]) {
			if (joinNodes.find(start) != joinNodes.end()) {
				combineNodeSubgraphs(agentSet, components, start, i);
				ISearch<> dijkstraSearch;
				SearchResult searchResult = dijkstraSearch.startSearch(map, agentSet, start.i, start.j, 0, 0,
																	   isGoal, true, true, 0, -1, m - 2, components[i]);
				while (searchResult.pathfound) {
					auto path = searchResult.lppath;
					for (auto it = std::next(path.begin()); std::next(it) != path.end(); ++it) {
						if (agentSet.getSubgraphs(it->i, it->j).empty()) {
							agentSet.setNodeSubgraph(it->i, it->j, i);
						}
					}
					combineNodeSubgraphs(agentSet, components, path.back(), i);
					searchResult = dijkstraSearch.startSearch(map, agentSet, start.i, start.j, 0, 0,
															  isGoal, false, m - 2);
				}
			}
		}
	}
}

int PushAndRotate::getReachableNodesCount(const SubMap &map, MAPFActorSet &agentSet, Node &start,
										  bool (*condition)(const Node &, const Node &, const SubMap &,
															const MAPFActorSet &),
										  const std::unordered_set<Node, NodeHash> &occupiedNodes) {
	int res = 0;
	ISearch<> dijkstraSearch;
	SearchResult searchResult = dijkstraSearch.startSearch(map, agentSet, start.i, start.j, 0, 0,
														   condition, true, false, 0, -1, -1, occupiedNodes);
	while (searchResult.pathfound) {
		++res;
		searchResult = dijkstraSearch.startSearch(map, agentSet, start.i, start.j, 0, 0,
												  condition, false, false, 0, -1, -1, occupiedNodes);
	}
	return res;
}

void PushAndRotate::assignToSubgraphs(const SubMap &map, MAPFActorSet &agentSet) {
	auto isUnoccupied = [](const Node &start, const Node &cur, const SubMap &map, const MAPFActorSet &agentSet) {
		return !agentSet.isOccupied(cur.i, cur.j);
	};

	std::vector<int> agentsInConnectedComponents(agentSet.getConnectedComponentsCount());
	for (int i = 0; i < agentSet.getActorCount(); ++i) {
		MAPFActor agent = agentSet.getActor(i);
		++agentsInConnectedComponents[agentSet.getConnectedComponent(agent.getCur_i(), agent.getCur_j())];
	}

	int m = map.GetEmptyCellCount() - agentSet.getActorCount();
	for (int i = 0; i < map.GetHeight(); ++i) {
		for (int j = 0; j < map.GetWidth(); ++j) {
			if (map.CellIsObstacle(i, j)) {
				continue;
			}
			Node pos(i, j);
			auto subgraphs = agentSet.getSubgraphs(pos.i, pos.j);
			if (subgraphs.empty()) {
				continue;
			}
			int subgraph = subgraphs[0];
			auto successors = search->findSuccessors(pos, map);
			int totalCount = agentSet.getComponentSize(pos.i, pos.j) -
							 agentsInConnectedComponents[agentSet.getConnectedComponent(pos.i, pos.j)];
			int throughPos = 0;
			bool hasSuccessorsInOtherSubgraph = false;
			for (auto neigh: successors) {
				auto neighSubgraphs = agentSet.getSubgraphs(neigh.i, neigh.j);
				if (neighSubgraphs.empty() || neighSubgraphs[0] != subgraph) {
					hasSuccessorsInOtherSubgraph = true;
					int throughNeigh = getReachableNodesCount(map, agentSet, neigh, isUnoccupied, {pos});
					int m1 = totalCount - throughNeigh;
					if (m1 >= 1 && m1 < m && agentSet.isOccupied(i, j)) {
						agentSet.setActorSubgraph(agentSet.getActorId(pos.i, pos.j), subgraph);
					}
					auto isGoal = [](const Node &start, const Node &cur, const SubMap &map,
									 const MAPFActorSet &agentSet) {
						return map.GetCellDegree(cur.i, cur.j) == 1 || !agentSet.getSubgraphs(cur.i, cur.j).empty();
					};
					ISearch<> dijkstraSearch;
					SearchResult searchResult = dijkstraSearch.startSearch(map, agentSet, neigh.i, neigh.j,
																		   0, 0, isGoal, true, true, 0, -1, -1, {pos});
					auto path = searchResult.lppath;
					int agentCount = 0;
					for (auto node: path) {
						if (agentSet.isOccupied(node.i, node.j)) {
							if (agentCount >= m1 - 1) {
								break;
							}
							agentSet.setActorSubgraph(agentSet.getActorId(node.i, node.j), subgraph);
							++agentCount;
						}
					}
					throughPos += throughNeigh;
				}
			}
			if (agentSet.isOccupied(i, j) && (!hasSuccessorsInOtherSubgraph || totalCount - throughPos >= 1)) {
				agentSet.setActorSubgraph(agentSet.getActorId(pos.i, pos.j), subgraph);
			}
		}
	}
}

void PushAndRotate::getPriorities(const SubMap &map, MAPFActorSet &agentSet) {
	std::unordered_map<Node, int, NodeHash> goalPositions;
	for (int i = 0; i < agentSet.getActorCount(); ++i) {
		goalPositions[agentSet.getActor(i).getGoalPosition()] = i;
	}

	auto isGoal = [](const Node &start, const Node &cur, const SubMap &map, const MAPFActorSet &agentSet) {
		return !agentSet.getSubgraphs(cur.i, cur.j).empty();
	};
	for (int i = 0; i < map.GetHeight(); ++i) {
		for (int j = 0; j < map.GetWidth(); ++j) {
			if (map.CellIsObstacle(i, j)) {
				continue;
			}

			auto subgraphs = agentSet.getSubgraphs(i, j);
			if (subgraphs.empty()) {
				continue;
			}
			int subgraph = subgraphs[0];
			auto successors = search->findSuccessors(Node(i, j), map);
			for (auto neigh: successors) {
				auto neighSubgraphs = agentSet.getSubgraphs(neigh.i, neigh.j);
				if (neighSubgraphs.empty() || neighSubgraphs[0] != subgraph) {
					ISearch<> dijkstraSearch;
					SearchResult searchResult = dijkstraSearch.startSearch(map, agentSet, neigh.i, neigh.j, 0, 0,
																		   isGoal, true, true, 0, -1, -1, {Node(i, j)});
					if (!searchResult.pathfound) {
						continue;
					}
					auto path = searchResult.lppath;
					path.push_front(Node(i, j));
					for (auto node: path) {
						auto it = goalPositions.find(node);
						if (it == goalPositions.end()) {
							break;
						}
						MAPFActor agent = agentSet.getActor(it->second);
						int agentSubgraph = agent.getSubgraph();
						if (agentSubgraph != -1) {
							if (agentSubgraph != subgraph) {
								agentSet.setPriority(subgraph, agentSubgraph);
							}
							break;
						}
					}
				}
			}
		}
	}
}


MAPFSearchResult PushAndRotate::startSearch(const SubMap &map, const MAPFConfig &config, MAPFActorSet &agentSet) {
	// //  std::cout << agentSet.getActorCount() << std::endl;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	getSubgraphs(map, agentSet);

	MAPFActorSet goalAgentSet = agentSet;
	for (int i = 0; i < agentSet.getActorCount(); ++i) {
		Node goal = agentSet.getActor(i).getGoalPosition();
		goalAgentSet.setActorPosition(i, goal);
	}
	assignToSubgraphs(map, agentSet);
	assignToSubgraphs(map, goalAgentSet);
	for (int i = 0; i < agentSet.getActorCount(); ++i) {
		if (agentSet.getActor(i).getSubgraph() != goalAgentSet.getActor(i).getSubgraph()) {
			result.pathfound = false;
			return result;
		}
	}
	getPriorities(map, agentSet);

	result.pathfound = solve(map, config, agentSet, begin);
	if (result.pathfound) {
		if (config.parallelizePaths1) {
			getParallelPaths(agentSet, config);
		}
		else {
			getPaths(agentSet);
		}
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	int elapsedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	if (elapsedMilliseconds > config.maxTime) {
		result.pathfound = false;
	}
	if (result.pathfound) {
		result.agentsMoves = new std::vector<ActorMove>(agentsMoves);
		result.agentsPaths = new std::vector<std::vector<Node>>(agentsPaths);
		result.time = static_cast<double>(elapsedMilliseconds);
	}
	else {
		result.agentsMoves = nullptr;
		result.agentsPaths = nullptr;
		result.time = static_cast<double>(elapsedMilliseconds);
	}
	return result;
}

PushAndRotate::PushAndRotate(const PushAndRotate &obj) : MAPFSearchInterface(obj) {
	search = obj.search;
	agentsMoves = obj.agentsMoves;
	result = obj.result;
}

PushAndRotate &PushAndRotate::operator=(const PushAndRotate &obj) {
	if (this != &obj) {
		MAPFSearchInterface::operator=(obj);
		search = obj.search;
		agentsMoves = obj.agentsMoves;
		result = obj.result;
	}
	return *this;
}
