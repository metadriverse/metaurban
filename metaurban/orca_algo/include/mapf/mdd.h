#include <vector>
#include <unordered_set>
#include "constraints.h"
#include "../sub_map.h"
#include "mapf_actor_set.h"
#include "isearch.h"
#include "astar.h"


#ifndef ORCASTAR_MDD_H
#define ORCASTAR_MDD_H


class MDD {
	public:
		MDD();

		template<typename SearchType>
		MDD(const SubMap &map, const MAPFActorSet &agentSet, SearchType *search, int agentId, int cost,
			const ConstraintsSet &constraints = ConstraintsSet());

		int getLayerSize(int cost) const;

//private:
		std::vector<int> layerSizes;
};

template<typename SearchType>
MDD::MDD(const SubMap &map, const MAPFActorSet &agentSet, SearchType *search, int agentId, int cost,
		 const ConstraintsSet &constraints) {
	Astar<> astar;
	MAPFActor agent = agentSet.getActor(agentId);
	Node start = agent.getStartPosition(), goal = agent.getGoalPosition();
	std::vector<std::unordered_set<Node, NodeHash>> layers;
	layers.push_back({start});

	int t = 0;
	int q = 0;

	for (int i = 0; i < cost - 1; ++i) {
		layers.push_back({});
		for (auto node: layers[i]) {

			std::chrono::steady_clock::time_point a = std::chrono::steady_clock::now();

			std::list<Node> successors = astar.findSuccessors(node, map, goal.i, goal.j, agentId, {}, constraints);

			std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
			q += std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
			++t;

			for (auto neigh: successors) {
				if (search->computeHFromCellToCell(neigh.i, neigh.j, goal.i, goal.j) <= cost - i - 1) {
					layers.back().insert(neigh);
				}
			}
		}
	}

	//std::cout << q << std::endl;
	//std::cout << t << std::endl;

	layerSizes.resize(cost + 1, 0);
	layerSizes[cost] = 1;
	std::unordered_set<Node, NodeHash> lastLayer = {goal};
	for (int i = cost - 1; i >= 0; --i) {
		std::unordered_set<Node, NodeHash> newLastLayer;
		for (auto node: layers[i]) {
			std::list<Node> successors = astar.findSuccessors(node, map, goal.i, goal.j, agentId, {}, constraints);
			for (auto neigh: successors) {
				if (lastLayer.find(neigh) != lastLayer.end()) {
					newLastLayer.insert(node);
					break;
				}
			}
		}
		layerSizes[i] = newLastLayer.size();
		lastLayer = newLastLayer;
	}
}

#endif //ORCASTAR_MDD_H
