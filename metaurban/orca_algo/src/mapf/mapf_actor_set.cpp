#include "mapf/mapf_actor_set.h"

void MAPFActorSet::clear() {
	occupiedNodes.clear();
	actors.clear();
}

void MAPFActorSet::addActor(int start_i, int start_j, int goal_i, int goal_j) {
	occupiedNodes[std::make_pair(start_i, start_j)] = actors.size();
	actors.push_back(MAPFActor(start_i, start_j, goal_i, goal_j, actors.size()));
}

void MAPFActorSet::setActorPosition(int actorId, Node pos) {
	actors[actorId].setCurPosition(pos.i, pos.j);
}

void MAPFActorSet::setPriority(int first, int second) {
	subgraphPriorities.insert(std::make_pair(first, second));
}

void MAPFActorSet::setConnectedComponent(int i, int j, int compNum) {
	connectivityComponents[std::make_pair(i, j)] = compNum;
}

void MAPFActorSet::addComponentSize(int compSize) {
	componentSizes.push_back(compSize);
}

int MAPFActorSet::getActorCount() const {
	return actors.size();
}

MAPFActor MAPFActorSet::getActor(int id) const {
	return actors[id];
}

bool MAPFActorSet::isOccupied(int i, int j) const {
	return occupiedNodes.find(std::make_pair(i, j)) != occupiedNodes.end();
}

int MAPFActorSet::getActorId(int i, int j) const {
	return occupiedNodes.at(std::make_pair(i, j));
}

void MAPFActorSet::moveActor(Node &from, Node &to, std::vector<ActorMove> &result) {
	if (from.i == to.i && from.j == to.j) {
		//  std::cout << "AAAA\n";
	}
	int id = occupiedNodes.at(std::make_pair(from.i, from.j));
	occupiedNodes[std::make_pair(to.i, to.j)] = id;
	occupiedNodes.erase(std::make_pair(from.i, from.j));
	actors[id].setCurPosition(to.i, to.j);
	result.push_back(ActorMove(to.i - from.i, to.j - from.j, id));
}


void MAPFActorSet::setNodeSubgraph(int i, int j, int subgraphNum) {
	subgraphNodes.insert(std::make_pair(std::make_pair(i, j), subgraphNum));
}

void MAPFActorSet::setActorSubgraph(int actorId, int subgraphNum) {
	actors[actorId].setSubgraph(subgraphNum);
}

void MAPFActorSet::removeSubgraphs(int i, int j) {
	subgraphNodes.erase(std::make_pair(i, j));
}

std::vector<int> MAPFActorSet::getSubgraphs(int i, int j) const {
	std::vector<int> res;
	std::pair<int, int> pair = std::make_pair(i, j);
	for (auto it = subgraphNodes.lower_bound(pair); it != subgraphNodes.upper_bound(pair); ++it) {
		res.push_back(it->second);
	}
	return res;
}

bool MAPFActorSet::hasPriority(int first, int second) const {
	return subgraphPriorities.find(std::make_pair(first, second)) != subgraphPriorities.end();
}

int MAPFActorSet::getConnectedComponentsCount() const {
	return componentSizes.size();
}

int MAPFActorSet::getComponentSize(int i, int j) {
	return componentSizes[connectivityComponents[std::make_pair(i, j)]];
}

int MAPFActorSet::getConnectedComponent(int i, int j) {
	return connectivityComponents[std::make_pair(i, j)];
}
