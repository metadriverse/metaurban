#include "agent.h"
#include <cmath>
#include <random>

#ifndef ORCA_ORCADDAGENT_H
#define ORCA_ORCADDAGENT_H


class ORCADDAgent : public Agent {
	public:
		ORCADDAgent();

		ORCADDAgent(const int &id, const Point &start, const Point &goal, const Map &map,
					const environment_options &options,
					AgentParam param, float effR, float wheelTrack, float theta);

		ORCADDAgent(const ORCADDAgent &obj);

		~ORCADDAgent();


		ORCADDAgent *Clone() const override;

		void ComputeNewVelocity() override;

		void ApplyNewVelocity() override;

		bool UpdatePrefVelocity() override;

		void SetPosition(const Point &pos) override;

		bool isFinished() override;

		bool operator==(const ORCADDAgent &another) const;

		bool operator!=(const ORCADDAgent &another) const;

		ORCADDAgent &operator=(const ORCADDAgent &obj);

	protected:
		void ComputeWheelsSpeed();

		Point effectivePosition;
		Point effectiveVelocity;
		float effectiveRadius;
		float wheelTrack;
		float D;

		float leftV;
		float rightV;
		float cos0;
		float sin0;
		float tet;


//    std::vector <std::pair<float, Agent*>> Neighbours;
//    std::vector <std::pair<float, ObstacleSegment>> NeighboursObst;
//
//    std::vector <Line> ORCALines;
//
//    Point position;
//    Velocity prefV;
//    Velocity newV;
//    Velocity currV;
//

};


#endif //ORCA_ORCADDAGENT_H
