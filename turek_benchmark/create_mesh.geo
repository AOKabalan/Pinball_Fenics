// Mesh resolution
res = 0.01;
coarse_res = 5*res;
fine_res = 0.5*res;

// Channel dimensions
L = 2.2;
H = 0.41;

// Obstacle parameters
c_x = 0.2;
c_y = 0.2;
r_x = 0.05;

// Channel points
Point(1) = {0, 0, 0, res};
Point(2) = {L, 0, 0, coarse_res};
Point(3) = {L, H, 0, coarse_res};
Point(4) = {0, H, 0, res};

// Obstacle center point
Point(5) = {c_x, c_y, 0, fine_res};

// Obstacle points
Point(6) = {c_x-r_x, c_y, 0, fine_res};    // Left
Point(7) = {c_x, c_y+r_x, 0, fine_res};    // Top
Point(8) = {c_x+r_x, c_y, 0, fine_res};    // Right
Point(9) = {c_x, c_y-r_x, 0, fine_res};    // Bottom

// Channel lines
Line(1) = {1, 2};    // Bottom
Line(2) = {2, 3};    // Right
Line(3) = {3, 4};    // Top
Line(4) = {4, 1};    // Left

// Obstacle elliptical arcs
Ellipse(5) = {6, 5, 7, 7};    // Left to top
Ellipse(6) = {7, 5, 8, 8};    // Top to right
Ellipse(7) = {8, 5, 9, 9};    // Right to bottom
Ellipse(8) = {9, 5, 6, 6};    // Bottom to left

// Create line loops
Line Loop(1) = {1, 2, 3, 4};          // Channel outer boundary
Line Loop(2) = {5, 6, 7, 8};          // Obstacle boundary

// Create surface with hole
Plane Surface(1) = {1, 2};

// Physical groups
Physical Surface(12) = {1};           // Domain
Physical Line(1) = {4};               // Inflow
Physical Line(2) = {2};               // Outflow
Physical Line(3) = {1, 3};            // Walls
Physical Line(4) = {5, 6, 7, 8};      // Obstacle

Mesh 2;
Save "turek_cylinder.msh";
