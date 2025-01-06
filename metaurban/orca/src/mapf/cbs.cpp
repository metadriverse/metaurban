#include "mapf/cbs.h"

int CBSNode::curId = 0;

template<typename SearchType>
ConflictBasedSearch<SearchType>::ConflictBasedSearch() {
	search = nullptr;
}

template<typename SearchType>
ConflictBasedSearch<SearchType>::ConflictBasedSearch(SearchType *Search) {
	search = Search;
}

template<typename SearchType>
ConflictBasedSearch<SearchType>::~ConflictBasedSearch() {
//    if (search)
//        delete search;
}

template<typename SearchType>
std::list<Node>
ConflictBasedSearch<SearchType>::getNewPath(const SubMap &map, const MAPFActorSet &agentSet, const MAPFActor &agent,
											const Constraint &constraint, const ConstraintsSet &constraints,
											const std::list<Node>::iterator pathStart,
											const std::list<Node>::iterator pathEnd,
											bool withCAT, const ConflictAvoidanceTable &CAT,
											std::vector<double> &lb,
											std::vector<int> &LLExpansions, std::vector<int> &LLNodes) {
	std::vector<Constraint> positiveConstraints = constraints.getPositiveConstraints();
	std::sort(positiveConstraints.begin(), positiveConstraints.end(),
			  [](const Constraint &lhs, const Constraint &rhs) {
				  return lhs.time < rhs.time || lhs.time == rhs.time && lhs.prev_i != -1 && rhs.prev_i == -1;
			  });
	int startTime = 0, endTime = -1;
	Node start = agent.getStartPosition(), end = agent.getGoalPosition();
	int i;
	for (i = 0; i < positiveConstraints.size(); ++i) {
		if (positiveConstraints[i].time >= constraint.time) {
			if (positiveConstraints[i].prev_i == -1) {
				end = Node(positiveConstraints[i].i, positiveConstraints[i].j);
				endTime = positiveConstraints[i].time;
			}
			else {
				if (positiveConstraints[i].time == constraint.time &&
					positiveConstraints[i].i == constraint.i && positiveConstraints[i].j == constraint.j &&
					(constraint.prev_i == -1 || constraint.prev_i == positiveConstraints[i].prev_i &&
												constraint.prev_j == positiveConstraints[i].prev_j)) {
					return std::list<Node>();
				}
				end = Node(positiveConstraints[i].prev_i, positiveConstraints[i].prev_j);
				endTime = positiveConstraints[i].time - 1;
			}
			if (i > 0) {
				start = Node(positiveConstraints[i - 1].i, positiveConstraints[i - 1].j);
				startTime = positiveConstraints[i - 1].time;
			}
			break;
		}
	}
	if (i == positiveConstraints.size() && i > 0) {
		start = Node(positiveConstraints[i - 1].i, positiveConstraints[i - 1].j);
		startTime = positiveConstraints[i - 1].time;
	}

	SearchResult searchResult = search->startSearch(map, agentSet, start.i, start.j, end.i, end.j, nullptr,
													true, true, startTime, endTime, -1, {}, constraints, withCAT, CAT);

	LLNodes.push_back(searchResult.nodescreated);
	LLExpansions.push_back(searchResult.nodesexpanded);
	if (!searchResult.pathfound) {
		return std::list<Node>();
	}

	double newLb = searchResult.minF;
	std::list<Node> res;
	auto it1 = searchResult.lppath.begin();
	int time = 0;
	for (auto it2 = pathStart; it2 != pathEnd && (endTime != -1 || it1 != searchResult.lppath.end()); ++it2) {
		if (time < startTime || (endTime != -1 && time > endTime)) {
			res.push_back(*it2);
			++newLb;
		}
		else {
			res.push_back(*it1);
			++it1;
		}
		++time;
	}
	for (; time < startTime; ++time) {
		res.push_back(*std::prev(pathEnd));
	}
	for (; it1 != searchResult.lppath.end(); ++it1) {
		res.push_back(*it1);
	}
	lb[agent.getId()] = newLb;
	return res;
}

template<typename SearchType>
CBSNode
ConflictBasedSearch<SearchType>::createNode(const SubMap &map, const MAPFActorSet &agentSet, const MAPFConfig &config,
											const Conflict &conflict, const std::vector<int> &costs,
											ConstraintsSet &constraints, int id1, int id2,
											const Node &pos1, const Node &pos2,
											std::vector<std::list<Node>::iterator> &starts,
											std::vector<std::list<Node>::iterator> &ends,
											ConflictAvoidanceTable &CAT, ConflictSet &conflictSet,
											std::vector<MDD> &mdds, std::vector<double> &lb,
											std::vector<int> &LLExpansions, std::vector<int> &LLNodes,
											CBSNode *parentPtr) {
	Constraint constraint(pos1.i, pos1.j, conflict.time, id1);
	if (conflict.edgeConflict) {
		constraint.prev_i = pos2.i;
		constraint.prev_j = pos2.j;
	}

	ConstraintsSet agentConstraints = constraints.getAgentConstraints(id1);
	agentConstraints.addConstraint(constraint);

	MAPFActor agent = agentSet.getActor(id1);
	if (config.withCAT || config.withFocalSearch == true) {
		CAT.removeAgentPath(starts[id1], ends[id1]);
	}

	double oldLb = lb[id1];
	std::list<Node> newPath = getNewPath(map, agentSet, agent, constraint, agentConstraints,
										 starts[id1], ends[id1], config.withCAT, CAT, lb, LLExpansions, LLNodes);

	if (config.withCAT || config.withFocalSearch == true) {
		CAT.addAgentPath(starts[id1], ends[id1]);
	}
	if (newPath.empty()) {
		return CBSNode(false);
	}

	CBSNode node;
	node.paths[id1] = newPath;
	for (int i = 0; i < costs.size(); ++i) {
		if (i == id1) {
			node.cost += newPath.size() - 1;
		}
		else {
			node.cost += costs[i];
		}
	}
	node.constraint = constraint;
	node.parent = parentPtr;

	MDD oldMDD;
	if (config.withCardinalConflicts) {
		node.mdds[id1] = MDD(map, agentSet, search, id1, newPath.size() - 1, agentConstraints);
		oldMDD = mdds[id1];
		mdds[id1] = node.mdds[id1];
	}

	auto oldStart = starts[id1], oldEnd = ends[id1];
	starts[id1] = node.paths[id1].begin();
	ends[id1] = node.paths[id1].end();
	if (config.storeConflicts) {
		ConflictSet agentConflicts = findConflict<std::list<Node>::iterator>(starts, ends, id1, true,
																			 config.withCardinalConflicts, mdds);
		node.conflictSet = conflictSet;
		node.conflictSet.replaceAgentConflicts(id1, agentConflicts);
	}
	if (config.withFocalSearch) {
		node.hc = node.conflictSet.getConflictingPairsCount();
		node.sumLb = 0;
		for (auto x: lb) {
			node.sumLb += x;
		}
		node.lb[id1] = lb[id1];
		lb[id1] = oldLb;
	}

	if (config.withMatchingHeuristic) {
		node.H = node.conflictSet.getMatchingHeuristic();
	}

	starts[id1] = oldStart;
	ends[id1] = oldEnd;
	if (config.withCardinalConflicts) {
		mdds[id1] = oldMDD;
	}

	node.G = node.H + node.cost;
	return node;
}

template<typename SearchType>
MAPFSearchResult
ConflictBasedSearch<SearchType>::startSearch(const SubMap &map, const MAPFConfig &config, MAPFActorSet &agentSet) {
	// //  std::cout << agentSet.getAgentCount() << std::endl;

	ISearch<>::T = 0;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	if (config.withPerfectHeuristic) {
		search->getPerfectHeuristic(map, agentSet);
	}

	auto focalCmp = [](const CBSNode &lhs, const CBSNode &rhs) {
		return lhs.hc < rhs.hc || lhs.hc == rhs.hc && lhs < rhs;
	};
	std::set<CBSNode, decltype(focalCmp)> focal(focalCmp);
	std::multiset<double> sumLb;

	CBSNode root;
	int agentCount = agentSet.getActorCount();
	std::vector<std::list<Node>::iterator> starts(agentCount), ends(agentCount);
	std::vector<MDD> mdds;
	for (int i = 0; i < agentSet.getActorCount(); ++i) {
		MAPFActor agent = agentSet.getActor(i);
		Astar<> astar(false, false);
		SearchResult searchResult = astar.startSearch(map, agentSet, agent.getStart_i(), agent.getStart_j(),
													  agent.getGoal_i(), agent.getGoal_j());
		if (!searchResult.pathfound) {
			//  std::cout << agent.getStart_i() << " " << agent.getStart_j() << " " << agent.getGoal_i() << " "
					//   << agent.getGoal_j() << "\n";
			//  std::cout << "fail" << std::endl;
		}
		root.cost += searchResult.pathlength;
		root.paths[i] = searchResult.lppath;
		// //  std::cout<<i << " " << root.paths[i].size() << "\n";

		if (config.withCardinalConflicts) {
			root.mdds[i] = MDD(map, agentSet, search, i, searchResult.pathlength);
		}

		starts[i] = root.paths[i].begin();
		ends[i] = root.paths[i].end();
		mdds.push_back(root.mdds[i]);
		if (config.withFocalSearch) {
			root.lb[i] = searchResult.minF;
			root.sumLb += searchResult.minF;
		}
	}

	if (config.storeConflicts) {
		root.conflictSet = findConflict<std::list<Node>::iterator>(starts, ends, -1, true,
																   config.withCardinalConflicts, mdds);
		if (config.withFocalSearch) {
			root.hc = root.conflictSet.getConflictingPairsCount();
			sumLb.insert(root.sumLb);
		}
	}

	MAPFSearchResult result(false);
	std::set<CBSNode> open = {root};
	std::list<CBSNode> close;
	std::vector<int> LLExpansions, LLNodes;

	int t = 0;
	while (!open.empty() || !focal.empty()) {
		++t;
		std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
		if (std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count() > config.maxTime) {
			result.pathfound = false;
			result.time = std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count();
			break;
		}

		CBSNode cur;

		if (config.withFocalSearch) {
			double threshold = *sumLb.begin() * config.focalW;
			auto it = open.begin();
			for (it; it != open.end() && it->cost <= threshold; ++it) {
				focal.insert(*it);
			}
			open.erase(open.begin(), it);
			cur = *focal.begin();
			focal.erase(cur);
			sumLb.erase(sumLb.find(cur.sumLb));
		}
		else {
			cur = *open.begin();
			open.erase(cur);
		}
		close.push_back(cur);

		std::vector<int> costs(agentCount, 0);
		std::vector<bool> agentFound(agentCount, false);
		std::vector<std::list<Node>::iterator> starts(agentCount), ends(agentCount);
		ConstraintsSet constraints;
		ConflictAvoidanceTable CAT;
		std::vector<MDD> mdds(agentCount);
		std::vector<double> lb(agentCount);
		for (CBSNode *ptr = &cur; ptr != nullptr; ptr = ptr->parent) {
			for (auto it = ptr->paths.begin(); it != ptr->paths.end(); ++it) {
				if (!agentFound[it->first]) {
					starts[it->first] = it->second.begin();
					ends[it->first] = it->second.end();
					costs[it->first] = it->second.size() - 1;
					agentFound[it->first] = true;

					if (config.withCAT || config.withFocalSearch) {
						CAT.addAgentPath(starts[it->first], ends[it->first]);
					}
					if (config.withCardinalConflicts) {
						mdds[it->first] = ptr->mdds.find(it->first)->second;
					}
					if (config.withFocalSearch) {
						lb[it->first] = ptr->lb[it->first];
					}
				}
			}
			if (ptr->id != 0) {
				constraints.addConstraint(ptr->constraint);
				if (ptr->hasPositiveConstraint) {
					constraints.addConstraint(ptr->positiveConstraint);
				}
			}
		}


		ConflictSet conflictSet;
		if (!config.storeConflicts) {
			conflictSet = findConflict<std::list<Node>::iterator>(starts, ends, -1, false,
																  config.withCardinalConflicts, mdds);
		}
		else {
			conflictSet = cur.conflictSet;
		}
		// conflictSet.PrintConflicts();

		if (conflictSet.empty()) {
			agentsPaths.resize(agentCount);
			for (int i = 0; i < agentCount; ++i) {
				for (auto it = starts[i]; it != ends[i]; ++it) {
					agentsPaths[i].push_back(*it);
				}
			}
			result.agentsPaths = new std::vector<std::vector<Node>>(agentsPaths);
			result.pathfound = true;
			result.HLExpansions = close.size();
			result.HLNodes = open.size() + close.size() + focal.size();
			if (LLExpansions.empty()) {
				result.AvgLLExpansions = 0;
				result.AvgLLNodes = 0;
			}
			else {
				result.AvgLLExpansions =
						(double) std::accumulate(LLExpansions.begin(), LLExpansions.end(), 0) / LLExpansions.size();
				result.AvgLLNodes = (double) std::accumulate(LLNodes.begin(), LLNodes.end(), 0) / LLNodes.size();
			}

			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			int elapsedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
			result.time = static_cast<double>(elapsedMilliseconds) / 1000;
			break;
		}

		Conflict conflict = conflictSet.getBestConflict();
		if (config.withDisjointSplitting &&
			mdds[conflict.id1].getLayerSize(conflict.time) < mdds[conflict.id2].getLayerSize(conflict.time)) {
			std::swap(conflict.id1, conflict.id2);
			std::swap(conflict.pos1, conflict.pos2);
		}

		std::vector<CBSNode> children;
		CBSNode child1 = createNode(map, agentSet, config, conflict, costs, constraints,
									conflict.id1, conflict.id2, conflict.pos2, conflict.pos1,
									starts, ends, CAT, conflictSet, mdds, lb, LLExpansions, LLNodes, &close.back());
		if (child1.pathFound) {
			children.push_back(child1);
		}
		CBSNode child2 = createNode(map, agentSet, config, conflict, costs, constraints,
									conflict.id2, conflict.id1, conflict.pos1, conflict.pos2,
									starts, ends, CAT, conflictSet, mdds, lb, LLExpansions, LLNodes, &close.back());
		if (child2.pathFound) {
			children.push_back(child2);
		}

		bool bypass = false;
		if (children.size() == 2) {
			for (auto child: children) {
				if (child.pathFound && config.withBypassing && cur.cost == child.cost &&
					conflictSet.getConflictCount() > child.conflictSet.getConflictCount()) {
					open.insert(child);
					bypass = true;
					break;
				}
			}
		}
		if (!bypass && !children.empty()) {
			if (config.withDisjointSplitting) {
				children[0].hasPositiveConstraint = true;
				int id1 = children[0].paths.begin()->first;
				int id2 = conflict.id1 == id1 ? conflict.id2 : conflict.id1;
				Constraint positiveConstraint;
				if (conflict.edgeConflict) {
					positiveConstraint = Constraint(children[0].constraint.prev_i, children[0].constraint.prev_j,
													conflict.time, id2, children[0].constraint.i,
													children[0].constraint.j);
				}
				else {
					positiveConstraint = Constraint(children[0].constraint.i, children[0].constraint.j, conflict.time,
													id2);
				}
				positiveConstraint.positive = true;
				children[0].positiveConstraint = positiveConstraint;
			}

			for (auto child: children) {
				open.insert(child);
				if (config.withFocalSearch) {
					sumLb.insert(child.sumLb);
				}
			}
		}

//        //  std::cout << t << "\n";
	}
	////  std::cout << close.size() + open.size() << std::endl;
	////  std::cout << ISearch<>::T << std::endl;
	return result;
}

//template <typename SearchType> void ConflictBasedSearch<SearchType>::clear()
//{
//    MAPFSearchInterface::clear();
//    SearchType
//}

template
class ConflictBasedSearch<Astar<>>;

template
class ConflictBasedSearch<FocalSearch<>>;

template
class ConflictBasedSearch<SCIPP<>>;
