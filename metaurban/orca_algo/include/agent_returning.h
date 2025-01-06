#include <set>
#include <algorithm>
#include <cstdlib>
#include <random>

#include "agent.h"

#ifndef ORCASTAR_ORCAAGENTWITHRETURNING_H
#define ORCASTAR_ORCAAGENTWITHRETURNING_H


class ORCAAgentWithReturning : public Agent {
	public:
		ORCAAgentWithReturning();

		ORCAAgentWithReturning(const int &id, const Point &start, const Point &goal, const Map &map,
							   const environment_options &options,
							   AgentParam param);

		ORCAAgentWithReturning(const ORCAAgentWithReturning &obj);

		~ORCAAgentWithReturning();

		ORCAAgentWithReturning &operator=(const ORCAAgentWithReturning &obj);

		ORCAAgentWithReturning *Clone() const override;

		bool operator==(const ORCAAgentWithReturning &another) const;

		bool operator!=(const ORCAAgentWithReturning &another) const;

		void ComputeNewVelocity() override;

		void ApplyNewVelocity() override;

		bool UpdatePrefVelocity() override;

		void AddNeighbour(Agent &neighbour, float distSq) override;


	private:
		std::vector<std::pair<float, Agent *>> &GetNeighbours();

		void SendRequestsForReturning();

		void ReturnTosPastPoint();

		float fakeRadius;
		std::mt19937 gen;


};


#endif //ORCASTAR_ORCAAGENTWITHRETURNING_H
