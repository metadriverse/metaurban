#include "sipp_node.h"
#include "fs_node.h"


#ifndef ORCASTAR_SCIPPNODE_H
#define ORCASTAR_SCIPPNODE_H


struct SCIPPNode : public SIPPNode, FSNode {
	SCIPPNode(int x = 0, int y = 0, Node *p = nullptr, int g_ = 0, int H_ = 0, int ConflictsCount = 0, int hc_ = 0) :
			Node(x, y, p, g_, H_, ConflictsCount),
			SIPPNode(x, y, p, g_, H_, ConflictsCount),
			FSNode(x, y, p, g_, H_, ConflictsCount, hc_) {}

	SCIPPNode(const Node &other) : Node(other) {}

};

#endif //ORCASTAR_SCIPPNODE_H
