#include <list>

#include "environment_options.h"
#include "map.h"


#ifndef ORCA_PATHPLANNER_H
#define ORCA_PATHPLANNER_H


class PathPlanner {
	public:
		PathPlanner(const PathPlanner &obj) : map(obj.map), options(obj.options), glStart(obj.glStart),
											  glGoal(obj.glGoal), radius(obj.radius) {}

		PathPlanner(const Map &map, const environment_options &options, const Point &start, const Point &goal,
					const float &radius)
				: map(&map), options(&options), glStart(start), glGoal(goal), radius(radius) {};

		virtual ~PathPlanner() {
			map = nullptr;
			options = nullptr;
		}

		virtual PathPlanner *Clone() const = 0;

		virtual bool CreateGlobalPath() = 0;

		virtual bool GetNext(const Point &curr, Point &next) = 0;

		virtual void AddPointToPath(Point p) = 0;

		virtual Point PullOutNext() = 0;

		virtual Point GetPastPoint() = 0;

		PathPlanner &operator=(const PathPlanner &obj) {
			map = obj.map;
			options = obj.options;
			glStart = obj.glStart;
			glGoal = obj.glGoal;
			radius = obj.radius;
			return *this;
		}

	protected:
		const Map *map;
		const environment_options *options;
		Point glStart;
		Point glGoal;
		float radius;

};

#endif //ORCA_PATHPLANNER_H
