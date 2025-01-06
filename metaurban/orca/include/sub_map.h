#include "map.h"
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <list>
#include <queue>

#ifndef ORCA_SUBMAP_H
#define ORCA_SUBMAP_H


class SubMap {
	public:
		SubMap();

		SubMap(const Map *map, Node mapOrigin, std::pair<int, int> mapSize, int divK);

		SubMap(const SubMap &obj);

		~SubMap();

		bool CellIsObstacle(int i, int j) const;

		bool CellOnGrid(int i, int j) const;

		bool CellIsTraversable(int i, int j) const;

		bool CellIsTraversable(int i, int j, const std::unordered_set<Node, NodeHash> &occupiedNodes) const;

		int GetHeight() const;

		int GetWidth() const;

		int GetEmptyCellCount() const;

		int GetCellDegree(int i, int j) const;

		float GetCellSize() const;

		bool PointBelongsToArea(const Point &point) const;

		Node GetClosestNode(const Point &point) const;

		Point GetPoint(const Node &node) const;

		Node FindUnoccupiedNode(Node start, std::unordered_map<int, Node> occupied);

		Node FindAccessibleNodeForGoal(Node start, Point goal, std::unordered_map<int, Node> occupied,
									   std::unordered_map<int, Node> undesirable);

		Node FindCloseToPointAvailableNode(Point pos, std::unordered_map<int, Node> occupied);


		SubMap &operator=(const SubMap &obj);


		// void printSubMap() {
		// 	std::cout << "\n";
		// 	for (int i = 0; i < size.first; i++) {
		// 		for (int j = 0; j < size.second; j++) {
		// 			std::cout << CellIsObstacle(i, j);
		// 		}
		// 		std::cout << "\n";
		// 	}
		// }

	private:
		float cellSize;
		Node origin;
		std::pair<int, int> size;
		int divKoef;
		const Map *fullMap;
		int emptyCells;


};


#endif //ORCA_SUBMAP_H
