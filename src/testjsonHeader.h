#ifndef TESTJSONHEADER_H
#define TESTJSONHEADER_H

//#define MSGPACK_USE_X3_PARSE
//#define MSGPACK_USE_DEFINE_MAP
//#define MSGPACK_USE_CPP03
#include<msgpack.hpp>

#include <sstream>
#include <fstream>
#include<iostream>
#include<string>

#include "vec.h"

#include "comonUtils.h"


void parseJson();


__host__ struct testClass {
	Vec3d a;
	std::vector<double> b;
#ifndef __CUDACC__
	MSGPACK_DEFINE_MAP(a,b);
#endif
	void test();
};

struct StdJoint {
	std::vector<int> left;// the indices of the left points
	std::vector<int> right;// the indices of the right points
	std::vector<int> anchor;// the 2 indices of the anchor points: left_anchor_id,right_anchor_id
	int leftCoord;
	int rightCoord;
	Vec3d axis;
#ifndef __CUDACC__ // not defined when compiling host code
	MSGPACK_DEFINE_MAP(left, right, anchor, leftCoord, rightCoord, axis);
#endif
};

class Model {
public:
	double radius_poisson;// poisson discretization radius
	std::vector<Vec3d> vertices;// the mass xyzs
	std::vector<Vec2i> edges;//the spring ids
	std::vector<Vec3i> triangles; // the triangle indices
	std::vector<bool> isSurface;// whether the mass is near the surface
	std::vector<int> idVertices;// the edge id of the vertices
	std::vector<int> idEdges;// the edge id of the springs
	std::vector<Vec3d> colors;// the mass xyzs
	std::vector<StdJoint> joints;// the joints
#ifndef __CUDACC__
	MSGPACK_DEFINE_MAP(radius_poisson, vertices, edges, triangles, isSurface, idVertices, idEdges, colors, joints) // write the member variables that you want to pack
#endif
	Model() {}
	Model(const std::string& file_path, bool versbose = true);
};



#endif //TESTJSONHEADER_H