#include "mapf/conflict_avoidance_table.h"

void ConflictAvoidanceTable::addNode(const Node &node) {
	auto tuple = std::make_tuple(node.i, node.j, node.g);
	if (nodeAgentsCount.find(tuple) == nodeAgentsCount.end()) {
		nodeAgentsCount[tuple] = 1;
	}
	else {
		++nodeAgentsCount[tuple];
	}
}

void ConflictAvoidanceTable::addEdge(const Node &node, const Node &prev) {
	auto tuple = std::make_tuple(prev.i, prev.j, node.i, node.j, node.g);
	if (edgeAgentsCount.find(tuple) == edgeAgentsCount.end()) {
		edgeAgentsCount[tuple] = 1;
	}
	else {
		++edgeAgentsCount[tuple];
	}
}

void ConflictAvoidanceTable::addAgentPath(const std::list<Node>::const_iterator &start,
										  const std::list<Node>::const_iterator &end) {
	for (auto it = start; it != end; ++it) {
		addNode(*it);
		if (it != start && *it != *std::prev(it)) {
			addEdge(*it, *std::prev(it));
		}
	}
}

void ConflictAvoidanceTable::removeNode(const Node &node) {
	auto tuple = std::make_tuple(node.i, node.j, node.g);
	if (nodeAgentsCount[tuple] == 1) {
		nodeAgentsCount.erase(tuple);
	}
	else {
		--nodeAgentsCount[tuple];
	}
}

void ConflictAvoidanceTable::removeEdge(const Node &node, const Node &prev) {
	auto tuple = std::make_tuple(prev.i, prev.j, node.i, node.j, node.g);
	if (edgeAgentsCount[tuple] == 1) {
		edgeAgentsCount.erase(tuple);
	}
	else {
		--edgeAgentsCount[tuple];
	}
}

void ConflictAvoidanceTable::removeAgentPath(const std::list<Node>::const_iterator &start,
											 const std::list<Node>::const_iterator &end) {
	for (auto it = start; it != end; ++it) {
		removeNode(*it);
		if (it != start && *it != *std::prev(it)) {
			removeEdge(*it, *std::prev(it));
		}
	}
}

int ConflictAvoidanceTable::getAgentsCount(const Node &node) const {
	int res = 0;
	auto tuple = std::make_tuple(node.i, node.j, node.g);
	if (nodeAgentsCount.find(tuple) != nodeAgentsCount.end()) {
		res = nodeAgentsCount.at(tuple);
	}
	return res;
}

int ConflictAvoidanceTable::getFirstSoftConflict(const Node &node, int startTime, int endTime) const {
	auto nodeIt = nodeAgentsCount.lower_bound(std::make_tuple(node.i, node.j, startTime));
	if (nodeIt != nodeAgentsCount.end() && std::get<0>(nodeIt->first) == node.i
		&& std::get<1>(nodeIt->first) == node.j && std::get<2>(nodeIt->first) <= endTime) {
		return std::get<2>(nodeIt->first);
	}
	return -1;
}

int ConflictAvoidanceTable::getFutureConflictsCount(const Node &node, int time) const {
	int res = 0;
	auto it = nodeAgentsCount.upper_bound(std::make_tuple(node.i, node.j, time));
	for (; it != nodeAgentsCount.end() && std::get<0>(it->first) == node.i
		   && std::get<1>(it->first) == node.j; ++it) {
		res += it->second;
	}
	return res;
}

void ConflictAvoidanceTable::getSoftConflictIntervals(std::vector<std::pair<int, int>> &res,
													  const Node &node, const Node &prevNode,
													  int startTime, int endTime, bool binary) const {
	std::map<int, int> agentsCount;
	auto nodeIt = nodeAgentsCount.lower_bound(std::make_tuple(node.i, node.j, startTime));
	auto nodeEnd = nodeAgentsCount.upper_bound(std::make_tuple(node.i, node.j, endTime));
	for (nodeIt; nodeIt != nodeEnd; ++nodeIt) {
		agentsCount[std::get<2>(nodeIt->first)] = nodeIt->second;
	}

	auto edgeIt = edgeAgentsCount.lower_bound(std::make_tuple(node.i, node.j, prevNode.i, prevNode.j, startTime));
	auto edgeEnd = edgeAgentsCount.upper_bound(std::make_tuple(node.i, node.j, prevNode.i, prevNode.j, endTime));
	for (edgeIt; edgeIt != edgeEnd; ++edgeIt) {
		int time = std::get<4>(edgeIt->first);
		if (agentsCount.find(time) == agentsCount.end()) {
			agentsCount[std::get<4>(edgeIt->first)] = 0;
		}
		agentsCount[std::get<4>(edgeIt->first)] += edgeIt->second;
	}

	int count = 0, prevTime = startTime - 1, beg = -1;
	for (auto it = agentsCount.begin(); it != agentsCount.end(); ++it) {
		int time = it->first;
		if (time > prevTime + 1 || count == 0 || (!binary && it->second != count)) {
			if (beg != -1) {
				res.push_back(std::make_pair(beg, count));
			}
			if (time > prevTime + 1) {
				res.push_back(std::make_pair(prevTime + 1, 0));
			}
			beg = time;
			count = it->second;
		}
		prevTime = time;
	}
	if (beg != -1) {
		res.push_back(std::make_pair(beg, count));
	}
	if (prevTime < endTime) {
		res.push_back(std::make_pair(prevTime + 1, 0));
	}
}
