#include "../geom.h"
#include "../sub_map.h"
#include <set>
#include <unordered_map>
#include "fs_node.h"
#include "scipp_node.h"

#ifndef ORCASTAR_SEARCHQUEUE_H
#define ORCASTAR_SEARCHQUEUE_H


template<typename NodeType = Node>
class SearchQueue {
	public:
		SearchQueue(bool (*cmp)(const NodeType &, const NodeType &) = [](const NodeType &lhs, const NodeType &rhs) {
			return lhs < rhs;
		});

		bool insert(const SubMap &map, NodeType node, bool withTime,
					bool withOld = false, NodeType old = NodeType(-1, -1));

		void erase(const SubMap &map, NodeType node, bool withTime);

		void moveByThreshold(SearchQueue<NodeType> &other, double threshold, const SubMap &map,
							 std::multiset<double> &otherF,
							 bool withTime = false);

		NodeType getByIndex(const SubMap &map, NodeType node, bool withTime);

		NodeType getFront() const;

		bool empty() const;

		int size() const;

		void clear();

//private:
		std::set<NodeType, bool (*)(const NodeType &, const NodeType &)> sortByKey;
		std::unordered_map<int, NodeType> sortByIndex;

		bool (*cmp)(const NodeType &, const NodeType &);
};


#endif //ORCASTAR_SEARCHQUEUE_H
