#include <easy3d/core/surface_mesh.h>
#include <easy3d/viewer/viewer.h>
#include <easy3d/util/initializer.h>
#include <easy3d/core/surface_mesh_builder.h>
 
using namespace easy3d;
namespace vhop_visualization_{
 
void visualize_mesh(const std::vector<vec3>& vertices, const std::vector<std::vector<int>>&faces)  {
    // Initialize Easy3D.
    initialize();
    easy3d::SurfaceMesh mesh;
    SurfaceMeshBuilder builder(&mesh);
    builder.begin_surface();
    Viewer viewer("Mesh ");//name of the opened window
    for (int f=0;f<faces.size(); f++){

   SurfaceMesh::Vertex v0 = builder.add_vertex(vertices[faces[f][0]]);
   SurfaceMesh::Vertex v1 = builder.add_vertex(vertices[faces[f][1]]);
   SurfaceMesh::Vertex v2 = builder.add_vertex(vertices[faces[f][2]]);
   builder.add_triangle(v0, v1, v2);//create triangles with vertices and faces
  }
   builder.end_surface(false);
   viewer.add_model(&mesh);
   viewer.fit_screen();
   viewer.run();

     
}
}
