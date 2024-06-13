#include <vector>
#include <algorithm>
#include <unordered_set>
#include <set>

#include "conflict.h"
#include "../geom.h"
#include <iostream>

#ifndef ORCASTAR_CONFLICTSET_H
#define ORCASTAR_CONFLICTSET_H


class ConflictSet {
	public:
		void addCardinalConflict(Conflict &conflict);

		void addSemiCardinalConflict(Conflict &conflict);

		void addNonCardinalConflict(Conflict &conflict);

		void replaceAgentConflicts(int agentId, ConflictSet &agentConflicts);

		bool empty();

		Conflict getBestConflict();

		int getConflictCount();

		int getCardinalConflictCount();

		std::vector<Conflict> getCardinalConflicts();

		int getMatchingHeuristic();

		int getConflictingPairsCount();

		void PrintConflicts() {
			for (int i = 0; i < nonCardinal.size(); i++) {
				std::cout << nonCardinal[i].id1 << " " << nonCardinal[i].id2 << " " << nonCardinal[i].time << "\n";
			}
		}

	private:
		std::vector<Conflict> cardinal, semiCardinal, nonCardinal;
};


#endif //ORCASTAR_CONFLICTSET_H
