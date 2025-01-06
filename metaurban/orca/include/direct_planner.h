#include "path_planner.h"
#include "geom.h"


#ifndef ORCA_DIRECTPLANNER_H
#define ORCA_DIRECTPLANNER_H


class DirectPlanner : public PathPlanner {
	public:
		DirectPlanner(const Map &map, const environment_options &options, const Point &start, const Point &goal,
					  const float &radius);

		DirectPlanner(const DirectPlanner &obj);

		~DirectPlanner() override;

		bool GetNext(const Point &curr, Point &next) override;

		bool CreateGlobalPath() override;

		DirectPlanner *Clone() const override;

		DirectPlanner &operator=(const DirectPlanner &obj);

		void AddPointToPath(Point p) override;

		Point PullOutNext() override;

		Point GetPastPoint() override;

};

#endif //ORCA_DIRECTPLANNER_H
