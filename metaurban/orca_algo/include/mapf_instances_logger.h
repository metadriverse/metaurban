


#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>

#include "mapf/mapf_actor_set.h"
#include "sub_map.h"
#include "mapf/mapf_config.h"
#include "mapf/mapf_search_result.h"

#include "tinyxml2.h"

#ifndef ORCASTAR_MAPFINSTANCESLOGGER_H
#define ORCASTAR_MAPFINSTANCESLOGGER_H

using namespace std;
using namespace tinyxml2;


class MAPFInstancesLogger {
	public:
		MAPFInstancesLogger() = default;

		~MAPFInstancesLogger();

		MAPFInstancesLogger(const MAPFInstancesLogger &obj);

		MAPFInstancesLogger(std::string pathTempl);

		MAPFInstancesLogger &operator=(const MAPFInstancesLogger &obj);


		bool SaveInstance(MAPFActorSet &agents, SubMap &map, MAPFConfig &conf);

//    void AddResults(const MAPFSearchResult &result);
		size_t GetFileID() const;


	private:
		size_t fileID;
		std::string pathTemplate;
//    std::string resPath;
//    basic_ofstream<char> pre_log;



};

#endif //ORCASTAR_MAPFINSTANCESLOGGER_H