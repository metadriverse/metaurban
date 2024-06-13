#include "map.h"

Map::Map() {
	cellSize = 1;
	height = 0;
	width = 0;
	grid = nullptr;
	obstacles = nullptr;
}


Map::Map(float cellSize, std::vector<std::vector<int>> &grid, std::vector<std::vector<Point>> &obstacles) {
	this->cellSize = cellSize;
	this->grid = new std::vector<std::vector<int>>(grid);
	this->obstacles = new std::vector<std::vector<ObstacleSegment>>();

	height = this->grid->size();
	width = 0;
	if (height) {
		width = (*this->grid)[0].size();
	}

	int idCounter = 0;

	for (auto obstacle: obstacles) {
		if (obstacle.size() < 2) {
			continue;
		}

		bool rCvx = true, lCvx;
		std::vector<ObstacleSegment> tmpObstacle;
		for (int i = 0; i < obstacle.size(); i++) {
			Vertex left = obstacle[(i == 0 ? obstacle.size() - 1 : i - 1)];
			Vertex right = obstacle[i];
			Vertex next = obstacle[(i == obstacle.size() - 1 ? 0 : i + 1)];
			lCvx = rCvx;
			rCvx = true;
			if (obstacle.size() > 2) {
				rCvx = (left - next).Det(right - left) >= 0.0f;
			}
			right.SetConvex(rCvx);
			left.SetConvex(lCvx);

			tmpObstacle.emplace_back(ObstacleSegment(idCounter, left, right));
			idCounter++;
		}
		tmpObstacle[0].left.SetConvex(rCvx);
		this->obstacles->push_back(tmpObstacle);
		tmpObstacle.clear();
	}

	for (auto &obstacle: (*this->obstacles)) {
		for (int i = 0; i < obstacle.size(); i++) {
			obstacle[i].next = (i == obstacle.size() - 1) ? &obstacle.at(0) : &obstacle.at(i + 1);
			obstacle[i].prev = (i == 0) ? &obstacle.at(obstacle.size() - 1) : &obstacle.at(i - 1);
		}
	}
}


Map::Map(const Map &obj) {
	cellSize = obj.cellSize;
	grid = (obj.grid == nullptr) ? nullptr : new std::vector<std::vector<int>>(*obj.grid);
	obstacles = (obj.obstacles == nullptr) ? nullptr : new std::vector<std::vector<ObstacleSegment>>(*obj.obstacles);
	height = obj.height;
	width = obj.width;
}


Map::~Map() {
	if (grid != nullptr) {
		delete grid;
		grid = nullptr;
	}

	if (obstacles != nullptr) {
		delete obstacles;
		obstacles = nullptr;
	}
}


bool Map::CellIsObstacle(int i, int j) const {
	return ((*grid)[i][j] != CN_GC_NOOBS);
}


bool Map::CellIsTraversable(int i, int j) const {
	return ((*grid)[i][j] == CN_GC_NOOBS);
}


bool Map::CellOnGrid(int i, int j) const {
	return (i < height && i >= 0 && j < width && j >= 0);
}


unsigned int Map::GetHeight() const {
	return height;
}


unsigned int Map::GetWidth() const {
	return width;
}


float Map::GetCellSize() const {
	return cellSize;
}


Node Map::GetClosestNode(const Point &point) const {
	Node res;
	res.i = height - 1 - (int) (point.Y() / cellSize);
	res.j = (int) (point.X() / cellSize);

	if (res.i < 0) {
		res.i = 0;
	}
	if (res.i > height - 1) {
		res.i = height - 1;
	}
	if (res.j < 0) {
		res.j = 0;
	}
	if (res.j > width - 1) {
		res.j = width - 1;
	}

	return res;
}


Point Map::GetPoint(const Node &node) const {
	return {(node.j * cellSize + cellSize / 2), (height - 1 - node.i) * cellSize + cellSize / 2};
}

const std::vector<std::vector<ObstacleSegment>> &Map::GetObstacles() const {
	return *obstacles;
}

Map &Map::operator=(const Map &obj) {
	if (this != &obj) {
		cellSize = obj.cellSize;
		height = obj.height;
		width = obj.width;

		if (grid != nullptr) {
			delete grid;
		}
		grid = (obj.grid == nullptr) ? nullptr : new std::vector<std::vector<int>>(*obj.grid);

		if (obstacles != nullptr) {
			delete obstacles;
		}
		obstacles = (obj.obstacles == nullptr) ? nullptr : new std::vector<std::vector<ObstacleSegment>>(
				*obj.obstacles);
	}
	return *this;
}









