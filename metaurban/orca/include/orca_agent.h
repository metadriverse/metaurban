#include "agent.h"

#ifndef ORCA_ORCAAGENT_H
#define ORCA_ORCAAGENT_H


class orca_agent : public Agent {

	public:
		orca_agent();

		orca_agent(const int &id, const Point &start, const Point &goal, const Map &map,
				   const environment_options &options,
				   AgentParam param);

		orca_agent(const orca_agent &obj);

		~orca_agent();


		orca_agent *Clone() const override;

		void ComputeNewVelocity() override;

		void ApplyNewVelocity() override;

		bool UpdatePrefVelocity() override;

		bool operator==(const orca_agent &another) const;

		bool operator!=(const orca_agent &another) const;

		orca_agent &operator=(const orca_agent &obj);

	private:
		float fakeRadius;
};


#endif //ORCA_ORCAAGENT_H
