

#ifndef ORCASTAR_MAPFCONFIG_H
#define ORCASTAR_MAPFCONFIG_H

struct MAPFConfig {
	public:
		MAPFConfig() = default;

		MAPFConfig(const MAPFConfig &orig) = default;

		~MAPFConfig() = default;


		int maxTime = 1000;
		bool withCAT = false;
		bool withPerfectHeuristic = false;
		bool parallelizePaths1 = false;
		bool parallelizePaths2 = false;
		bool withCardinalConflicts = false;
		bool withBypassing = false;
		bool withMatchingHeuristic = false;
		bool storeConflicts = false;
		bool withDisjointSplitting;
		bool withFocalSearch = false;
		double focalW = ECBS_SUBOUT_FACTOR;
		std::string planner = "push_and_rotate";
		std::string lowLevel = "astar";

};

#endif //ORCASTAR_MAPFCONFIG_H


