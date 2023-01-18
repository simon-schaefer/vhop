 
#include <easy3d/viewer/viewer.h>
#include <easy3d/renderer/camera.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/core/types.h>
#include <easy3d/util/resource.h>
#include <easy3d/util/initializer.h>
 
 
using namespace easy3d;
namespace vhop{
 
void visualize_mesh(const std::vector<vec3>& vertices, const std::vector<std::vector<unsigned int>>& faces)  {
    // Initialize Easy3D.
    initialize();

    Viewer viewer("Tutorial_301_Drawables");
    auto surface = new TrianglesDrawable("faces");
    surface->update_vertex_buffer(vertices);
    surface->update_element_buffer(faces);
    viewer.add_drawable(surface);
    viewer.fit_screen();
     viewer.run();

     
}
}