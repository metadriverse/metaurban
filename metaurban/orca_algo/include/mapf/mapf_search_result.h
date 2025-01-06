#include "../geom.h"


#ifndef ORCASTAR_MAPFSEARCHRESULT_H
#define ORCASTAR_MAPFSEARCHRESULT_H


struct MAPFSearchResult {
	bool pathfound;
	std::vector<ActorMove> *agentsMoves = nullptr;
	std::vector<std::vector<Node>> *agentsPaths = nullptr;
	double time;
	double AvgLLExpansions = 0;
	double AvgLLNodes = 0;
	int HLExpansions = 0;
	int HLNodes = 0;

	~MAPFSearchResult() {
		if (agentsMoves != nullptr) {
			delete agentsMoves;
		}
		if (agentsPaths != nullptr) {
			delete agentsPaths;
		}
	}

	MAPFSearchResult(bool Pathfound = false) {
		pathfound = Pathfound;
	}

	MAPFSearchResult &operator=(const MAPFSearchResult &obj) {
		if (&obj != this) {
			if (agentsMoves != nullptr) {
				delete agentsMoves;
			}
			if (agentsPaths != nullptr) {
				delete agentsPaths;
			}

			this->pathfound = obj.pathfound;
			this->agentsMoves = (obj.agentsMoves == nullptr) ? nullptr : new std::vector<ActorMove>(*(obj.agentsMoves));
			this->agentsPaths = (obj.agentsPaths == nullptr) ? nullptr : new std::vector<std::vector<Node>>(
					*(obj.agentsPaths));
			this->time = obj.time;
			this->AvgLLExpansions = obj.AvgLLExpansions;
			this->AvgLLNodes = obj.AvgLLNodes;
			this->HLExpansions = obj.HLExpansions;
			this->HLNodes = obj.HLNodes;
		}
		return *this;
	}

	MAPFSearchResult(const MAPFSearchResult &obj) {
		this->pathfound = obj.pathfound;
		this->agentsMoves = (obj.agentsMoves == nullptr) ? nullptr : new std::vector<ActorMove>(*(obj.agentsMoves));
		this->agentsPaths = (obj.agentsPaths == nullptr) ? nullptr : new std::vector<std::vector<Node>>(
				*(obj.agentsPaths));
		this->time = obj.time;
		this->AvgLLExpansions = obj.AvgLLExpansions;
		this->AvgLLNodes = obj.AvgLLNodes;
		this->HLExpansions = obj.HLExpansions;
		this->HLNodes = obj.HLNodes;
	}

	void Clear() {
		pathfound = false;
		if (agentsMoves != nullptr) {
			delete agentsMoves;
			agentsMoves = nullptr;
		}
		if (agentsPaths != nullptr) {
			delete agentsPaths;
			agentsPaths = nullptr;
		}

		time = 0;
		AvgLLExpansions = 0;
		AvgLLNodes = 0;
		HLExpansions = 0;
		HLNodes = 0;
	}
};


#endif //ORCASTAR_MAPFSEARCHRESULT_H
