#ifndef ORCA_MISSION_H
#define ORCA_MISSION_H

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <chrono>

#include "agent.h"
#include "summary.h"
#include "xml_reader.h"
#include "xml_logger.h"
#include "mapf_instances_logger.h"

using namespace tinyxml2;

struct trajectory_dict {
	vector<float> xr;
	vector<float> yr;
	vector<float> nextxr;
	vector<float> nextyr;
	bool foundpath;
	int total_step;
};
class Mission {
	public:
		Mission() = delete;

		Mission(std::string fileName, unsigned int agentsNum, unsigned int stepsTh, bool time, size_t timeTh,
				bool speedStop);

		Mission(const Mission &obj);

		~Mission();

		bool ReadTask();

		Summary StartMission();

		Mission &operator=(const Mission &obj);

#if FULL_LOG

		bool SaveLog();
		unordered_map<int, trajectory_dict> SaveLog_dict();

#endif

	private:
		void UpdateSate();

		void AssignNeighbours();

		bool IsFinished();

		vector<Agent *> agents;
		Reader *taskReader;
		Map *map;
		environment_options *options;
		Summary missionResult;
		std::unordered_map<int, std::pair<bool, int>> resultsLog;
		MAPFInstancesLogger MAPFLog;

		bool isTimeBounded;
		bool stopByMeanSpeed;
		bool allStops;
		size_t timeTreshhold;

		unsigned int stepsCount;
		unsigned int stepsTreshhold;
		unsigned int agentsNum;
		unsigned int collisionsCount;
		unsigned int collisionsObstCount;
		std::vector<std::list<float>> commonSpeedsBuffer;


#if FULL_LOG
		Logger *taskLogger;
		std::unordered_map<int, std::vector<Point>> stepsLog;
		std::unordered_map<int, std::vector<Point>> goalsLog;
#endif
};


#endif //ORCA_MISSION_H
