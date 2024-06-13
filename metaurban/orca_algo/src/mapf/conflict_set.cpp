#include "mapf/conflict_set.h"

void ConflictSet::addCardinalConflict(Conflict &conflict) {
	cardinal.push_back(conflict);
}

void ConflictSet::addSemiCardinalConflict(Conflict &conflict) {
	semiCardinal.push_back(conflict);
}

void ConflictSet::addNonCardinalConflict(Conflict &conflict) {
	nonCardinal.push_back(conflict);
}

void ConflictSet::replaceAgentConflicts(int agentId, ConflictSet &agentConflicts) {
	std::vector<Conflict> newCardinal, newSemiCardinal, newNonCardinal;
	auto pred = [agentId](Conflict conflict) { return conflict.id1 != agentId && conflict.id2 != agentId; };
	std::copy_if(cardinal.begin(), cardinal.end(), std::back_inserter(newCardinal), pred);
	std::copy_if(semiCardinal.begin(), semiCardinal.end(), std::back_inserter(newSemiCardinal), pred);
	std::copy_if(nonCardinal.begin(), nonCardinal.end(), std::back_inserter(newNonCardinal), pred);
	newCardinal.insert(newCardinal.end(), agentConflicts.cardinal.begin(), agentConflicts.cardinal.end());
	newSemiCardinal.insert(newSemiCardinal.end(), agentConflicts.semiCardinal.begin(),
						   agentConflicts.semiCardinal.end());
	newNonCardinal.insert(newNonCardinal.end(), agentConflicts.nonCardinal.begin(), agentConflicts.nonCardinal.end());
	cardinal = newCardinal;
	semiCardinal = newSemiCardinal;
	nonCardinal = newNonCardinal;
}

bool ConflictSet::empty() {
	return cardinal.empty() && semiCardinal.empty() && nonCardinal.empty();
}

Conflict ConflictSet::getBestConflict() {
	if (!cardinal.empty()) {
		return cardinal[0];
	}
	else if (!semiCardinal.empty()) {
		return semiCardinal[0];
	}
	return nonCardinal[0];
}

int ConflictSet::getCardinalConflictCount() {
	return cardinal.size();
}

int ConflictSet::getConflictCount() {
	return cardinal.size() + semiCardinal.size() + nonCardinal.size();
}

std::vector<Conflict> ConflictSet::getCardinalConflicts() {
	return cardinal;
}

int ConflictSet::getMatchingHeuristic() {
	std::unordered_set<int> matched;
	int res = 0;
	for (auto conflict: cardinal) {
		if (matched.find(conflict.id1) == matched.end() && matched.find(conflict.id2) == matched.end()) {
			++res;
			matched.insert(conflict.id1);
			matched.insert(conflict.id2);
		}
	}
	return res;
}

int ConflictSet::getConflictingPairsCount() {
	std::set<std::pair<int, int>> conflictingPairs;
	std::vector<std::vector<Conflict>::iterator> begins = {nonCardinal.begin(), semiCardinal.begin(), cardinal.begin()};
	std::vector<std::vector<Conflict>::iterator> ends = {nonCardinal.end(), semiCardinal.end(), cardinal.end()};
	for (int i = 0; i < 3; ++i) {
		for (auto it = begins[i]; it != ends[i]; ++it) {
			auto pair = std::make_pair(it->id1, it->id2);
			if (pair.first > pair.second) {
				std::swap(pair.first, pair.second);
			}
			conflictingPairs.insert(std::make_pair(it->id1, it->id2));
		}
	}
	return conflictingPairs.size();
}


