#include <list>
#include <unordered_map>
#include <iostream>
#include <limits>

#include "path_planner.h"
#include "geom.h"
#include "line_of_sight.h"

#ifndef ORCA_THETASTAR_H
#define ORCA_THETASTAR_H


class ThetaStar : public PathPlanner {
	public:
		ThetaStar(const Map &map, const environment_options &options, const Point &start, const Point &goal,
				  const float &radius);

		ThetaStar(const ThetaStar &obj);

		~ThetaStar() override;

		bool GetNext(const Point &curr, Point &next) override;

		bool CreateGlobalPath() override;

		void AddPointToPath(Point p) override;

		Point PullOutNext() override;

		ThetaStar *Clone() const override;

		Point GetPastPoint() override;

		ThetaStar &operator=(const ThetaStar &obj);

	private:
		bool SearchPath(const Node &start, const Node &goal);

		bool StopCriterion() const;

		void AddOpen(Node newNode);

		Node FindMin();

		float ComputeHFromCellToCell(int i1, int j1, int i2, int j2) const;

		float Distance(int i1, int j1, int i2, int j2) const;

		Node ResetParent(Node current, Node parent);

		void MakePrimaryPath(Node curNode);

		std::list<Node> FindSuccessors(Node curNode);


		std::list<Point> currPath;
		Point past;
		std::unordered_map<int, Node> close;
		std::vector<std::list<Node>> open;
		int openSize;
		bool glPathCreated;
		LineOfSight visChecker;


};


#endif //ORCA_THETASTAR_H
