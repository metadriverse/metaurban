#include <list>
#include <vector>
#include <algorithm>
#include <chrono>

#include "isearch.h"
#include "mapf_actor_set.h"
#include "mapf_config.h"
#include "mapf_search_interface.h"
#include "mapf_search_result.h"
#include "../geom.h"
#include "../const.h"

#ifndef ORCA_PUSHANDROTATE_H
#define ORCA_PUSHANDROTATE_H


class PushAndRotate : public MAPFSearchInterface {
	public:
		PushAndRotate();

		PushAndRotate(ISearch<> *Search);

		~PushAndRotate(void);

		PushAndRotate(const PushAndRotate &obj);

		PushAndRotate &operator=(const PushAndRotate &obj);

		MAPFSearchResult startSearch(const SubMap &Map, const MAPFConfig &config, MAPFActorSet &AgentSet) override;

		void clear() override;

	private:
		bool solve(const SubMap &map, const MAPFConfig &config, MAPFActorSet &AgentSet,
				   std::chrono::steady_clock::time_point start);

		bool clearNode(const SubMap &map, MAPFActorSet &agentSet, Node &nodeToClear,
					   const std::unordered_set<Node, NodeHash> &occupiedNodes);

		bool push(const SubMap &map, MAPFActorSet &agentSet, Node &from, Node &to,
				  std::unordered_set<Node, NodeHash> &occupiedNodes);

		bool
		multipush(const SubMap &map, MAPFActorSet &agentSet, Node first, Node second, Node &to, std::list<Node> &path);

		bool clear(const SubMap &map, MAPFActorSet &agentSet, Node &first, Node &second);

		void exchange(const SubMap &map, MAPFActorSet &agentSet, Node &first, Node &second);

		void reverse(int begSize, int endSize,
					 int firstAgentId, int secondAgentId, MAPFActorSet &agentSet);

		bool swap(const SubMap &map, MAPFActorSet &agentSet, Node &first, Node &second);

		bool rotate(const SubMap &map, MAPFActorSet &agentSet, std::vector<Node> &qPath, int cycleBeg);

		void getPaths(MAPFActorSet &agentSet);

		void getParallelPaths(MAPFActorSet &agentSet, const MAPFConfig &config);

		void getComponent(MAPFActorSet &agentSet, std::pair<Node, Node> &startEdge,
						  std::vector<std::pair<Node, Node>> &edgeStack,
						  std::vector<std::unordered_set<Node, NodeHash>> &components);

		void combineNodeSubgraphs(MAPFActorSet &agentSet, std::vector<std::unordered_set<Node, NodeHash>> &components,
								  Node &subgraphNode, int subgraphNum);

		void getSubgraphs(const SubMap &map, MAPFActorSet &agentSet);

		int getReachableNodesCount(const SubMap &map, MAPFActorSet &agentSet, Node &start,
								   bool (*condition)(const Node &, const Node &, const SubMap &, const MAPFActorSet &),
								   const std::unordered_set<Node, NodeHash> &occupiedNodes);

		void assignToSubgraphs(const SubMap &map, MAPFActorSet &agentSet);

		void getPriorities(const SubMap &map, MAPFActorSet &agentSet);


		ISearch<> *search;
		std::vector<ActorMove> agentsMoves;
		MAPFSearchResult result;
};


#endif //ORCA_PUSHANDROTATE_H
