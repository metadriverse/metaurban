
#include "mapf_actor.h"

#include <vector>
#include <map>
#include <list>
#include <set>
#include <iostream>

#ifndef ORCA_PARACTORSET_H
#define ORCA_PARACTORSET_H


class MAPFActorSet {
	private:
		int width;
		int height;
		std::map<std::pair<int, int>, int> occupiedNodes;
		std::map<std::pair<int, int>, int> connectivityComponents;
		std::vector<int> componentSizes;
		std::multimap<std::pair<int, int>, int> subgraphNodes;
		std::set<std::pair<int, int>> subgraphPriorities;
		std::vector<MAPFActor> actors;

	public:
		void clear();

		void addActor(int start_i, int start_j, int goal_i, int goal_j);

		void moveActor(Node &from, Node &to, std::vector<ActorMove> &result);

		void setActorPosition(int actorId, Node pos);

		void setPriority(int first, int second);

		void setActorSubgraph(int actorId, int subgraphNum);

		void setConnectedComponent(int i, int j, int compNum);

		void addComponentSize(int compSize);

		void setNodeSubgraph(int i, int j, int subgraphNum);

		void removeSubgraphs(int i, int j);

		int getActorCount() const;

		MAPFActor getActor(int id) const;

		int getActorId(int i, int j) const;

		bool isOccupied(int i, int j) const;

		bool hasPriority(int first, int second) const;

		std::vector<int> getSubgraphs(int i, int j) const;

		int getConnectedComponentsCount() const;

		int getConnectedComponent(int i, int j);

		int getComponentSize(int i, int j);
};


#endif //ORCA_PARACTORSET_H
