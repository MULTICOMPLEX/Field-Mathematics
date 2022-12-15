
//Wolfram: PolyhedronData["ObtuseGoldenRhombohedron", "VertexCoordinates"]
//shape = PolyhedronData["ObtuseGoldenRhombohedron"]
//Export["shape.PLY", shape, "BinaryFormat" -> False]

//volume = 1/5 sqrt(10 - 2 sqrt(5))≈0.470228
//surface area = 12/sqrt(5)≈5.36656
//(assuming unit edge length)

//PolyhedronData[{"RhombicTriacontahedron", "C5"}, "InertiaTensor"] // TraditionalForm
//PolyhedronData["ObtuseGoldenRhombohedron", "InertiaTensor"] // TraditionalForm

#include <iostream> 
#include <fstream>    
#include <vector>


struct vertex {
	double x;
	double y;
	double z;
};

void createIcosahedron(std::vector<vertex> &);
void PLYExport(std::ofstream & stlout, const std::vector<vertex> & vertices);

const double OGR_v1 =  1/2. * sqrt(1/15. * (5 - 2. * sqrt(5))); //0.0937962
const double OGR_v2 =  1/2. * sqrt(3 -       6./sqrt(5));       //0.281389
const double OGR_v3 =  sqrt(1/30. * (5 + sqrt(5)));             //0.491123
const double OGR_v4 =  sqrt(1/10. * (5 + sqrt(5)));             //0.850651
const double OGR_v5 =  sqrt(2/15. * (5 + sqrt(5)));             //0.982247

std::vector <vertex> Obtuse_Golden_Rhombohedron
{
{0, 0, -0.2813887000083923},
{0.4911234676837921, -0.8506507873535156, -0.09379623830318451},
{-0.4911234676837921, -0.8506507873535156, 0.09379623830318451},
{-0.9822469353675842, 0, -0.09379623830318451},
{-0.4911234676837921, 0.8506507873535156, 0.09379623830318451},
{0, 0, 0.2813887000083923},
{0.9822469353675842, 0, 0.09379623830318451},
{0.4911234676837921, 0.8506507873535156, -0.09379623830318451}
};

//Wolfram: PolyhedronData["AcuteGoldenRhombohedron", "VertexCoordinates"]
//shape = PolyhedronData["AcuteGoldenRhombohedron"]
//Export["shape.PLY", shape, "BinaryFormat" -> False]

//volume = 1/5 sqrt(2 (5 + sqrt(5)))≈0.760845
//surface area = 12/sqrt(5)≈5.36656
//(assuming unit edge length)

const double AGR_v1 =  sqrt(3/4.  + 3./(2 * sqrt(5))); //1.19198
const double AGR_v2 =  sqrt(2/15. *    (5 - sqrt(5))); //0.607062
const double AGR_v3 =  sqrt(1/10. *    (5 - sqrt(5))); //0.525731
const double AGR_v4 =  sqrt(1/12. + 1./(6 * sqrt(5))); //0.397327
const double AGR_v5 =  sqrt(1/30. *    (5 - sqrt(5))); //0.303531

std::vector <vertex> Acute_Golden_Rhombohedron {
{0, 0, 1.191981673240662},
{-0.3035309910774231, 0.525731086730957, 0.3973272442817688},
{-0.6070619821548462, 0, -0.3973272442817688},
{-0.3035309910774231, -0.525731086730957, 0.3973272442817688},
{0.3035309910774231, -0.525731086730957, -0.3973272442817688},
{0, 0, -1.191981673240662},
{0.3035309910774231, 0.525731086730957, -0.3973272442817688},
{0.6070619821548462, 0, 0.3973272442817688}
};

void plot_vertices(std::vector<vertex>& v)
{
for (size_t i = 0; i < v.size(); i++) 
  {
    std::cout << v[i].x   << ", ";
    std::cout << v[i].y << ", ";
    std::cout << v[i].z;
    if(i < v.size() -1 )std::cout << ",\n";
  }
}

void PLYFacetOut(std::ofstream & stlout, const std::vector<vertex>& v) {

	// Export an PLY facet to an output stream, stlout.
	for (size_t i = 0; i < v.size(); i++) 
    stlout << v[i].x << " " << v[i].y << " " << v[i].z << std::endl;
}

void create_Golden_Rhombohedron(std::string name, const std::vector<vertex> & vertices)
{
  std::ofstream plyout(name, std::ofstream::trunc);
	// Export the data in text-based PLY format.
	PLYExport(plyout, vertices);

	plyout.close();
}

//Computer Numerical Control (CNC) 
//CNC router wood machining
//3D: STL, OBJ, DXF, STEP, IGES, DWG, and 3DM

void PLYExport(std::ofstream & stlout, const std::vector<vertex> & vertices) {
	// Export in PLY format to an output stream, stlout.
  
  stlout << "ply" << std::endl;
  stlout << "format ascii 1.0" << std::endl;
  stlout << "comment Created with C++" << std::endl;
  stlout << "element vertex 8" << std::endl;
  stlout << "property float x" << std::endl;
  stlout << "property float y" << std::endl;
  stlout << "property float z" << std::endl;
  stlout << "element face 6" << std::endl;
  stlout << "property list uchar int vertex_index" << std::endl;
  stlout << "end_header" << std::endl;
  
  PLYFacetOut(stlout, vertices);

 	stlout << "4 0 1 2 3" << std::endl;
	stlout << "4 4 5 6 7" << std::endl;
	stlout << "4 3 4 7 0" << std::endl;
	stlout << "4 1 6 5 2" << std::endl;
	stlout << "4 0 7 6 1" << std::endl;
	stlout << "4 2 5 4 3" << std::endl;

}