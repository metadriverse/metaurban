#include "const.h"
#include "path_planner.h"
#include "thetastar.h"
#include "geom.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <cstddef>
#include <list>

#ifndef ORCA_AGENT_H
#define ORCA_AGENT_H


class AgentParam {
	public:
		AgentParam() : sightRadius(CN_DEFAULT_RADIUS_OF_SIGHT), timeBoundary(CN_DEFAULT_TIME_BOUNDARY),
					   timeBoundaryObst(CN_DEFAULT_OBS_TIME_BOUNDARY),
					   radius(CN_DEFAULT_SIZE), maxSpeed(CN_DEFAULT_MAX_SPEED), agentsMaxNum(CN_DEFAULT_AGENTS_MAX_NUM),
					   rEps(CN_DEFAULT_REPS) {}

		AgentParam(float sr, float tb, float tbo, float r, float reps, float ms, int amn, int parNum, MAPFTriggers trig)
				: sightRadius(sr), timeBoundary(tb), timeBoundaryObst(tbo), radius(r), rEps(reps),
				  maxSpeed(ms), agentsMaxNum(amn) {}

		~AgentParam() = default;

		float sightRadius;
		float timeBoundary;
		float timeBoundaryObst;
		float radius;
		float rEps;
		float maxSpeed;
		int agentsMaxNum;
};

class Agent {
	public:
		Agent();

		Agent(const int &id, const Point &start, const Point &goal, const Map &map, const environment_options &options,
			  AgentParam param);

		Agent(const Agent &obj);

		virtual ~Agent();

		virtual Agent *Clone() const = 0;

		virtual void ComputeNewVelocity() = 0;

		virtual void ApplyNewVelocity() = 0;

		virtual bool UpdatePrefVelocity() = 0;


		virtual void SetPosition(const Point &pos);

		virtual void AddNeighbour(Agent &neighbour, float distSq);

		virtual bool isFinished();

		bool InitPath();

		int GetID() const;

		Point GetPosition() const;

		Point GetVelocity() const;

		float GetRadius() const;

		Point GetNext() const;


		std::pair<unsigned int, unsigned int> GetCollision() const;

		void UpdateNeighbourObst();

		bool operator==(const Agent &another) const;

		bool operator!=(const Agent &another) const;

		Agent &operator=(const Agent &obj);

		template<class Planner>
		void SetPlanner(const Planner &pl) {
			static_assert(std::is_base_of<PathPlanner, Planner>::value, "Planner should be inheritor of PathPlanner");
			this->planner = new Planner(pl);
		}


	protected:
		// bool MeanSpeedMAPFTrigger();
		bool CommonPointMAPFTrigger(float dist);

		bool NeighbourGroupMeanSpeedMAPFTrigger();

		bool SingleNeighbourMeanSpeedMAPFTrigger();

		int id;
		Point start;
		Point goal;
		PathPlanner *planner;
		const environment_options *options;
		const Map *map;
		std::vector<std::pair<float, Agent *>> Neighbours;
		std::vector<std::pair<float, ObstacleSegment>> NeighboursObst;

		std::vector<Line> ORCALines;

		Point position;
		Velocity prefV;
		Velocity newV;
		Velocity currV;

		AgentParam param;
		float invTimeBoundaryObst;
		float invTimeBoundary;
		float maxSqObstDist;

		unsigned int collisions;
		unsigned int collisionsObst;

		Point nextForLog;
		std::list<float> speedSaveBuffer;
		float meanSavedSpeed;


};


#endif //ORCA_AGENT_H
