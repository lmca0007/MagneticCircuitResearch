SetFactory("OpenCASCADE");

// -------------------------
// Geometry
// -------------------------

// Air box
air_size = 0.5;
Box(1) = {-air_size/2, -air_size/2, -air_size/2,
           air_size, air_size, air_size};

// Cylinder magnet
magnet_r = 0.02;
magnet_h = 0.04;
Cylinder(2) = {0, 0, -magnet_h/2, 0, 0, magnet_h, magnet_r};

// Boolean fragment: split box and cylinder into disjoint volumes
BooleanFragments{ Volume{1}; }{ Volume{2}; }

// Now the model contains *two new volumes* (IDs can vary!)
// Let’s fetch them programmatically:
v() = Volume "*"; // collects all volume tags

// Assign Physicals
Physical Volume("air")    = {v(0)};
Physical Volume("magnet") = {v(1)};

// -------------------------
// Mesh size control
// -------------------------

lc_fine   = 0.002;
lc_medium = 0.01;
lc_coarse = 0.1;

// Distance field from magnet surface (not volume!)
Field[1] = Distance;
Field[1].SurfacesList = {v(1)}; // boundary of magnet volume

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc_fine;
Field[2].LcMax = lc_medium;
Field[2].DistMin = 0.03;
Field[2].DistMax = 0.1;

Background Field = 2;

Mesh.Optimize = 1;
