#include "testjsonHeader.h"



//#include "simdjson.h"
//using namespace simdjson;
//
//
//void parseJson() {
//    ondemand::parser parser;
//    padded_string json = padded_string::load("../twitter.json");
//    ondemand::document tweets = parser.iterate(json);
//    //std::cout << uint64_t(tweets["search_metadata"]["count"]) << " results." << std::endl;
//}


//struct StdJoint_Msgpack:public StdJoint{
//public:
//	MSGPACK_DEFINE_ARRAY(left, right, anchor, leftCoord, rightCoord, axis);
//};

//class Model_Msgpack : public Model {
//public:
//	//std::vector<StdJoint_Msgpack> joints;// the joints
//MSGPACK_DEFINE_ARRAY(radius_poisson, vertices, edges, triangles, isSurface, idVertices, idEdges, colors, joints) // write the member variables that you want to pack
//};

Model::Model(const std::string& file_path, bool versbose) {
	// get the msgpack robot model
	// Deserialize the serialized data
	std::ifstream ifs(file_path, std::ifstream::in | std::ifstream::binary);
	std::stringstream buffer;
	buffer << ifs.rdbuf();
	msgpack::unpacked upd;//unpacked data
	msgpack::unpack(upd, buffer.str().data(), buffer.str().size());
	//    std::cout << upd.get() << std::endl;
	*this = (upd.get().as<Model>());
	if (versbose) {
		printf("Loaded %s\n", file_path.c_str());
		printf("radius_poisson=%.3e [m] \n", radius_poisson);
		printf("#vertices=%d, #edges=%d, #triangles = %d, #joints = %d\n", 
			(int)vertices.size(), (int)edges.size(), (int)triangles.size(), (int)joints.size());

	}
}

void testClass::test() {
	this->b.resize(30);
	for (int i = 0; i < 30; i++)
	{
		this->b[i] = i;
	}
	this->a.x=6;

	std::stringstream ss;
	msgpack::pack(ss, *this);

	auto const& str = ss.str();
	auto oh = msgpack::unpack(str.data(), str.size());
	auto obj = oh.get();
	std::cout << obj << std::endl;
}